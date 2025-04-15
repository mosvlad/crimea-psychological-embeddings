import pandas as pd
import numpy as np

from sklearn.manifold import TSNE

from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

import re
from string import punctuation

import hdbscan
import json
import os
from pathlib import Path
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache


CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def sanitize_filename(filename):
    """Creates a safe filename"""
    safe_filename = re.sub(r'[^\w\s-]', '', filename)
    safe_filename = re.sub(r'[-\s]+', '_', safe_filename)
    return safe_filename[:50]


def load_data(path):
    """Load data from CSV file"""
    return pd.read_csv(path, encoding='utf-8')


def get_cache_path(texts, prefix):
    """Generate cache path based on content hash"""
    # Sanitize the prefix first
    safe_prefix = re.sub(r'[^\w\s-]', '', prefix)
    safe_prefix = re.sub(r'[-\s]+', '_', safe_prefix)
    content_hash = hashlib.md5(''.join(texts).encode()).hexdigest()
    return CACHE_DIR / f"{safe_prefix}_{content_hash}.pkl"


@lru_cache(maxsize=1000)
def preprocess_text(text):
    """Cached text preprocessing"""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = ''.join([char for char in text if char not in punctuation])
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class TextEmbedder:
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts, cache_prefix='embeddings'):
        """Get embeddings with caching"""
        cache_path = get_cache_path(texts, cache_prefix)

        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        # Batch processing for better performance
        batch_size = 32
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)

        embeddings = np.array(embeddings)

        # Cache results
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f)

        return embeddings


def find_similar_answers(series, embedder):
    """Find similar answers using modern embeddings"""
    # Get embeddings
    texts = [preprocess_text(str(x)) for x in series.fillna('')]
    cache_prefix = sanitize_filename(f'similar_answers_{series.name}')
    embeddings = embedder.get_embeddings(texts, cache_prefix)

    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # Find groups of similar answers
    similar_groups = []
    used_indices = set()

    for i in range(len(series)):
        if i in used_indices:
            continue

        group = [i]
        used_indices.add(i)

        for j in range(i + 1, len(series)):
            if j not in used_indices and similarity_matrix[i, j] > 0.8:
                group.append(j)
                used_indices.add(j)

        if len(group) > 1:
            similar_groups.append([series.iloc[idx] for idx in group])

    return similar_groups


def save_cluster_info(column_name, clusters, original_data):
    """Save cluster information to JSON"""
    cluster_info = {}

    for cluster_id in set(clusters):
        if cluster_id == -1:  # Skip noise points
            continue

        # Get indices for this cluster
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_data = original_data.iloc[cluster_indices]

        cluster_info[f"cluster_{cluster_id}"] = {
            "size": len(cluster_indices),
            "percentage": len(cluster_indices) / len(clusters) * 100,
            "examples": cluster_data.head(5).tolist()
        }

    # Save to JSON
    safe_name = sanitize_filename(column_name)
    output_path = RESULTS_DIR / f"cluster_info_{safe_name}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_info, f, ensure_ascii=False, indent=2)


def visualize_text_clusters(column_name, clusters, embeddings, original_data):
    """Visualize text clusters using t-SNE"""
    try:
        # Apply t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            n_iter=1000,
            random_state=42
        )
        coords = tsne.fit_transform(embeddings)

        # Create DataFrame for visualization
        viz_df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'cluster': clusters,
            'original_text': original_data
        })

        # Create interactive plot
        fig = px.scatter(
            viz_df,
            x='x',
            y='y',
            color='cluster',
            hover_data=['original_text'],
            title=f'Answer Clusters for {column_name} (t-SNE visualization)'
        )

        # Improve layout
        fig.update_layout(
            width=1000,
            height=800,
            showlegend=True
        )

        # Save with safe filename
        safe_name = sanitize_filename(column_name)
        output_path = RESULTS_DIR / f'clusters_{safe_name}.html'
        fig.write_html(str(output_path))

    except Exception as e:
        print(f"Error visualizing clusters for {column_name}: {str(e)}")


def analyze_text_patterns(df, embedder):
    """Optimized text pattern analysis"""
    results = {}

    for column in df.columns:
        if df[column].dtype == 'object':
            texts = df[column].apply(preprocess_text).tolist()

            try:
                # Get cached embeddings
                cache_prefix = sanitize_filename(f'embeddings_{column}')
                embeddings = embedder.get_embeddings(texts, cache_prefix)

                # Optimized HDBSCAN parameters
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=5,
                    min_samples=3,
                    metric='euclidean',
                    cluster_selection_method='eom',
                    core_dist_n_jobs=-1  # Use all CPU cores
                )
                clusters = clusterer.fit_predict(embeddings)

                # Save cluster information
                save_cluster_info(column, clusters, df[column])

                results[column] = {
                    'clusters': clusters,
                    'embeddings': embeddings
                }

            except Exception as e:
                print(f"Skipping analysis for column {column}: {str(e)}")
                continue

    return results


def analyze_semantic_patterns(df, embedder):
    """Analyze semantic patterns in answers with caching"""
    patterns = {}

    for col in df.columns:
        if df[col].dtype == 'object':
            texts = df[col].apply(preprocess_text).tolist()

            try:
                # Use cached embeddings
                cache_prefix = sanitize_filename(f'semantic_{col}')
                embeddings = embedder.get_embeddings(texts, cache_prefix)

                # Create BERTopic model
                topic_model = BERTopic(language="multilingual")
                topics, _ = topic_model.fit_transform(texts)

                # Get topic info
                topic_info = topic_model.get_topic_info()

                # Save topic information to JSON
                safe_name = sanitize_filename(col)
                topic_path = RESULTS_DIR / f'topics_{safe_name}.json'
                topic_info.to_json(str(topic_path), orient='records', force_ascii=False)

                patterns[col] = {
                    'topics': topics,
                    'topic_info': topic_info
                }

            except Exception as e:
                print(f"Error analyzing patterns for {col}: {str(e)}")
                continue

    return patterns


def main():
    # Load data
    df = load_data("./data_8-11.csv")

    print("Loaded rows:", len(df))
    print("\nColumns in data:")
    for col in df.columns:
        print(f"- {col}")

    # Initialize embedder
    embedder = TextEmbedder()

    # Analyze text patterns with improved clustering
    text_patterns = analyze_text_patterns(df, embedder)

    # Parallel processing for visualizations
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Parallel cluster visualization
        visualization_tasks = [
            (column, results['clusters'], results['embeddings'], df[column])
            for column, results in text_patterns.items()
        ]
        executor.map(lambda x: visualize_text_clusters(*x), visualization_tasks)

    # Find similar answers
    print("\nAnalyzing similar answers:")
    similar_answers_results = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            similar_groups = find_similar_answers(df[column], embedder)
            if similar_groups:
                similar_answers_results[column] = [
                    {"group": idx, "answers": group}
                    for idx, group in enumerate(similar_groups)
                ]

    # Save similar answers to JSON
    with open(RESULTS_DIR / 'similar_answers.json', 'w', encoding='utf-8') as f:
        json.dump(similar_answers_results, f, ensure_ascii=False, indent=2)

    # Analyze semantic patterns
    patterns = analyze_semantic_patterns(df, embedder)
    print("\nSemantic patterns analysis complete. Check results directory for topic information.")

    print("\nAnalysis complete! Check the following directories:")
    print(f"1. {CACHE_DIR} - cached embeddings")
    print(f"2. {RESULTS_DIR} - results and visualizations:")


if __name__ == "__main__":
    main()
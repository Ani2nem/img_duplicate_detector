from image_similarity import load_images_from_folder, extract_features
from image_similarity import create_vocabulary, extract_bow_histogram
from image_similarity import cluster_images, group_images_by_cluster


def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python main.py <folder_path>")
        return

    folder_path = sys.argv[1]
    images, filenames = load_images_from_folder(folder_path)
    all_descriptors, descriptors_list = extract_features(images)
    kmeans = create_vocabulary(all_descriptors)
    histograms = extract_bow_histogram(descriptors_list, kmeans)
    labels = cluster_images(histograms)
    clusters = group_images_by_cluster(labels, filenames)

    # Separate out the cluster labeled -1
    regular_clusters = {k: v for k, v in clusters.items() if k != -1}
    noise_cluster = clusters.get(-1, [])

    for label, files in sorted(regular_clusters.items()):
        print(f"Cluster {label}: {files}")

    if noise_cluster:
        print("\n" * 2)  # Add some new lines to separate out the noise cluster
        print(f"Cluster -1 (Noise/Outliers): {noise_cluster}")


if __name__ == "__main__":
    main()

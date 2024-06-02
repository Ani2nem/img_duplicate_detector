from image_similarity import load_images_from_folder, extract_features
from image_similarity import create_vocabulary, extract_bow_histogram
from image_similarity import cluster_images, group_images_by_cluster
from image_similarity import print_clusters, save_clusters


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
    print_clusters(clusters)
    save_clusters(clusters, folder_path)


if __name__ == "__main__":
    main()

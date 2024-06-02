import cv2
import numpy as np
from sklearn.cluster import DBSCAN, MiniBatchKMeans
from os import listdir, makedirs, remove
from os.path import isfile, join, exists
from multiprocessing import Pool, cpu_count
import shutil


def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in listdir(folder):
        if isfile(join(folder, filename)):
            img = cv2.imread(join(folder, filename))
            if img is not None:
                img = cv2.resize(img, (300, 300))
                images.append(img)
                filenames.append(filename)
    return images, filenames


def extract_features_from_image(img):
    orb = cv2.ORB_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return descriptors


def extract_features(images):
    with Pool(cpu_count()) as pool:
        descriptors_list = pool.map(extract_features_from_image, images)
    all_descriptors = [desc for descriptors in descriptors_list
                       if descriptors is not None for desc in descriptors]
    return np.array(all_descriptors), descriptors_list


def create_vocabulary(all_descriptors, k=100):
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
    kmeans.fit(all_descriptors)
    return kmeans


def extract_bow_histogram(descriptors_list, kmeans):
    histograms = []
    for descriptors in descriptors_list:
        if descriptors is None or descriptors.size == 0:
            histograms.append(np.zeros(kmeans.n_clusters))
        else:
            words = kmeans.predict(descriptors)
            histogram, _ = np.histogram(words,
                                        bins=np.arange(kmeans.n_clusters + 1))
            histograms.append(histogram)
    return histograms


def cluster_images(histograms, eps=5, min_samples=2):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(histograms)
    labels = db.labels_
    return labels


def group_images_by_cluster(labels, filenames):
    clusters = {}
    for label, filename in zip(labels, filenames):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(filename)
    return clusters


def print_clusters(clusters):
    # Separate out the cluster labeled -1
    regular_clusters = {k: v for k, v in clusters.items() if k != -1}
    noise_cluster = clusters.get(-1, [])

    for label, files in sorted(regular_clusters.items()):
        print(f"\nCluster {label}: {files}")

    if noise_cluster:
        print("\n" * 2)  # Add some new lines to separate out the noise cluster
        print(f"Distinct Images (Noise/Outliers): {noise_cluster}\n")


def save_clusters(clusters, folder):
    # Separate out the cluster labeled -1
    regular_clusters = {k: v for k, v in clusters.items() if k != -1}
    noise_cluster = clusters.get(-1, [])

    # Create directories
    distinct_folder = join(folder, 'Distinct Images')
    duplicates_folder = join(folder, 'Duplicates')

    if not exists(distinct_folder):
        makedirs(distinct_folder)
    if not exists(duplicates_folder):
        makedirs(duplicates_folder)

    # Save distinct images
    for filename in noise_cluster:
        source_file = join(folder, filename)
        dest_file = join(distinct_folder, filename)
        shutil.move(source_file, dest_file)

    # Save all duplicate images to a single folder
    for label, files in sorted(regular_clusters.items()):
        for filename in files:
            source_file = join(folder, filename)
            dest_file = join(duplicates_folder, filename)
            shutil.move(source_file, dest_file)

    # Remove any remaining files in the folder
    for filename in listdir(folder):
        file_path = join(folder, filename)
        if isfile(file_path):
            remove(file_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python image_similarity.py <folder_path>")
        sys.exit(1)

    folder = sys.argv[1]
    images, filenames = load_images_from_folder(folder)
    all_descriptors, descriptors_list = extract_features(images)
    kmeans = create_vocabulary(all_descriptors)
    histograms = extract_bow_histogram(descriptors_list, kmeans)
    labels = cluster_images(histograms)
    clusters = group_images_by_cluster(labels, filenames)
    print_clusters(clusters)
    save_clusters(clusters, folder)

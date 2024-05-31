import cv2
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from os import listdir
from os.path import isfile, join


def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in listdir(folder):
        if isfile(join(folder, filename)):
            img = cv2.imread(join(folder, filename))
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames


def extract_features(images):
    sift = cv2.SIFT_create()
    all_descriptors = []
    descriptors_list = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is not None:
            all_descriptors.extend(descriptors)
            descriptors_list.append(descriptors)
        else:
            descriptors_list.append(np.array([]))  # handle img w/o descriptors
    return np.array(all_descriptors), descriptors_list


def create_vocabulary(all_descriptors, k=100):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(all_descriptors)
    return kmeans


def extract_bow_histogram(descriptors_list, kmeans):
    histograms = []
    for descriptors in descriptors_list:
        if descriptors.size == 0:
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


if __name__ == "__main__":
    import sys
    folder = sys.argv[1]
    images, filenames = load_images_from_folder(folder)
    all_descriptors, descriptors_list = extract_features(images)
    kmeans = create_vocabulary(all_descriptors)
    histograms = extract_bow_histogram(descriptors_list, kmeans)
    labels = cluster_images(histograms)
    clusters = group_images_by_cluster(labels, filenames)
    for label, files in clusters.items():
        print(f"Cluster {label}: {files}")

from nilearn import input_data, datasets, connectome
import pandas as pd
import numpy as np
import os
import scipy.io as sio

#Define folder structure
root_folder = "C:\\Users\\Yashas\\Documents\\ABIDE\\"
data_folder = "D:\\abide_processed\\ABIDEII-EMC_1\\ABIDEII-EMC_1\\"
phenotypic_file = root_folder + "ABIDEII_Composite_Phenotypic.csv"

#Compute functional connectivity matrix of a single subject
def compute_single_connectivity(timeseries):
    correlation_measure = connectome.ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([timeseries])[0]
    np.fill_diagonal(correlation_matrix, 0)

    return correlation_matrix

#Get the full timeseries for a list of subjects
def get_timeseries(subject_ids, atlas='msdl'):
    atlas = datasets.fetch_atlas_msdl()
    atlas_file = atlas['maps']
    labels = atlas['labels']

    masker = input_data.NiftiMapsMasker(maps_img=atlas_file, standardize=True, memory='nilearn_cache', verbose=5)
    timeseries = []
    subjects = get_filepaths(subject_ids)

    for subject in subjects:
        timeseries.append(masker.fit_transform(subject))

    return timeseries

#Get the full timeseries for a single subject
def get_single_timeseries(subject_id, atlas='msdl'):
    atlas = datasets.fetch_atlas_msdl()
    atlas_file = atlas['maps']
    labels = atlas['labels']

    masker = input_data.NiftiMapsMasker(maps_img=atlas_file, standardize=True, memory='nilearn_cache', verbose=5)
    subject = get_filepaths([subject_id])[0]

    timeseries = masker.fit_transform(subject)

    return timeseries

#Get phenotypic metrics for a list of subjects
def get_phenotypic(subject_ids, metric):
    df = pd.read_csv(phenotypic_file, encoding='latin1')
    sub_ids = df['SUB_ID'].tolist()
    values = df[metric].tolist()

    metrics = {}

    for i, sub in enumerate(sub_ids):
        if str(sub) in subject_ids:
            metrics[sub] = values[i]

    return metrics

#Construct filepaths to imaging files for a list of subjects
def get_filepaths(subject_ids, type='rest'):
    paths = []
    for id in subject_ids:
        if type == 'rest':
            paths.append(data_folder + id + "\\session_1\\rest_1\\swarrest.nii")
        elif type == 'anat':
            paths.append(data_folder + id + "\\session_1\\anat_1\\ranat.nii")
        else:
            raise ValueError("recieved invalid type")

    return paths

def get_subject_ids():
    return os.listdir(path=data_folder)

def prepare_data(features):
    for i, data in enumerate(features):
        if np.isnan(np.min(data)):
            features = np.delete(features, i, 0)

    split = round(0.8 * features.shape[0])

    train = features[:split]
    test = features[split:]

    return (train, test)

"""Adapted from Parisot et. al."""

# Compute vectorised connectivity networks to be used as feature vector
def vectorise_networks(networks):

    idx = np.triu_indices_from(networks[0], 1)
    norm_networks = [np.arctanh(mat) for mat in networks]
    vec_networks = [mat[idx] for mat in norm_networks]
    all_vec_networks = np.vstack(vec_networks)

    return all_vec_networks

#Compute functional connectivity matrix of multiple subjects
def compute_all_connectivity(subject_ids, atlas='msdl', save=False):
    all_networks = []

    for sub in subject_ids:
        timeseries = get_single_timeseries(sub, atlas=atlas)
        connectivity = compute_single_connectivity(timeseries)
        all_networks.append(connectivity)

        if save:
            subject_file = os.path.join(root_folder + 'networks\\', sub + '_' + 'connectivity.mat')
            sio.savemat(subject_file, {'connectivity': connectivity})

    return all_networks

# Load precomputed connectivity networks
def get_saved_networks(subject_ids):
    all_networks = []

    for sub in subject_ids:
        subject_file = os.path.join(root_folder + 'networks\\', sub + '_' + 'connectivity.mat')
        matrix = sio.loadmat(subject_file)['connectivity']
        all_networks.append(matrix)

    return all_networks

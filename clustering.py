import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
#from sklearn.preprocessing import StandardScaler
import numpy as np
# Suppress warnings. Comment this out if you wish to see the warning messages
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.cluster import HDBSCAN
import astropy.units as u
import astropy.coordinates as apycoords
import os
import glob
# from zero_point import zpt
from tqdm import tqdm


####################

r_min = 3000 # pc
r_max = 4000 # pc
shell = 1


############### CLUSTERING #########

def clustering(df, min_cluster, min_samples, metric='mahalanobis', method='leaf', allow_single_cluster=True, \
               save_members=False, path_to_save='../data/', clustering_on=['ra']):
    
    print('')
    print('Performing clustering on {} stars using {}.'.format(len(df), clustering_on))
    print('Hyperparameters: min_cluster_size = {} - min_samples = {}'.format(min_cluster, min_samples))
    print('Metric {}'.format(metric))
    print('')
    
    data = df[clustering_on]
    data = RobustScaler().fit_transform(data)
    #data = StandardScaler().fit(data).transform(data)

    if metric == 'mahalanobis':
        clusterer = HDBSCAN(min_cluster_size=min_cluster, min_samples=min_samples, metric='mahalanobis', cluster_selection_method=method, \
                            allow_single_cluster=allow_single_cluster, metric_params={'V': np.cov(data, rowvar=False)}, n_jobs=-1)
    elif metric == 'euclidean':
        clusterer = HDBSCAN(min_cluster_size=min_cluster, min_samples=min_samples, metric=metric, cluster_selection_method=method, \
                            allow_single_cluster=allow_single_cluster, n_jobs=-1)
    else:
        print('Metric not selected!')
        print('Stopping code execution!')
        exit()
    
    clusterer.fit(data)
    labels = clusterer.labels_
    df['labels'] = labels
    unique_labels = set(labels)

    print('')
    print('{} clusters found.'.format(len(unique_labels)-1))
    print('{} stars were clasified as noise.'.format(len(df[(df['labels'] == -1)])))
    print('')
    
    for i in unique_labels:
        
        if i != -1:
            df_test = df[(df['labels'] == i)]
            filename = 'cluster_{}.csv'.format(i)

            if save_members:
                df_test.to_csv(os.path.join(path_to_save, filename))


############# READ DATA ###########

data = pd.read_csv('../data/field_2_error_{}_to_{}_pc.csv'.format(r_min, r_max))

sc = apycoords.SkyCoord(ra=data['ra'].values *u.deg, dec=data['dec'].values *u.deg, \
                        pm_ra_cosdec=data['pmra'].values *u.mas/u.yr, pm_dec=data['pmdec'].values*u.mas/u.yr)

data['pm_l'] = sc.galactic.pm_l_cosb.value
data['pm_b'] = sc.galactic.pm_b.value

k = 4.74047
data['v_alpha'] = k * data['pmra'] / data['parallax'] # km/s
data['v_delta'] = k * data['pmdec'] / data['parallax'] # km/s

print('')
print(data.info())
print('')
print(data.keys())
print('')

#clustering_on=['l','b','pm_l','pm_b','parallax']
clustering_on = ['l','b','pmra','pmdec','parallax']
#clustering_on=['ra','dec','pmra','pmdec','parallax']

############# QUADRANTS ############
# There are 12/8 quadrants

#healpix_label = np.unique(data['gaia_healpix_5'])
#partitions = np.linspace(min(healpix_label), max(healpix_label), num=13, dtype=int)
partitions = np.array([0., 45., 90., 135., 180., 225., 270., 315., 360.])

print('')
print('Partitions: ', partitions)
print('')

#quadrant = 
print('Shell from {} pc to {} pc'.format(r_min, r_max))


#for quadrant in tqdm(range(1,13)):
#for quadrant in tqdm(range(7,9)):
for quadrant in tqdm(range(8,9)):
    print('Quadrant number %s'%quadrant)

    # if quadrant != 12:
    #     df = data[(data['gaia_healpix_5'] >= partitions[quadrant-1])&(data['gaia_healpix_5'] < partitions[quadrant])].reset_index(drop=True)
    # else:
    #     df = data[(data['gaia_healpix_5'] >= partitions[quadrant-1])&(data['gaia_healpix_5'] <= partitions[quadrant])].reset_index(drop=True)

    df = data[(data['l'] >= partitions[quadrant-1]) & (data['l'] <= partitions[quadrant])].reset_index(drop=True)
    
    print('')
    print('Properties quadrant {}'.format(quadrant))
    print('')
    print(df.info())
    print('')

    ############ REMOVE RESULTS #########

    remove_previous_results = True

    if remove_previous_results:

        # remove previous csv results
        dir_path = '../data_new_clusters/shell{}/clusters{}/'.format(shell, quadrant)
        file_pattern = "*.csv"
        file_paths = glob.glob(os.path.join(dir_path, file_pattern))
        for file_path in file_paths:
            os.remove(file_path)

        # remove previous image results
        dir_path = '../data_new_clusters/images{}/Q{}/'.format(shell, quadrant)
        file_pattern = "*.png"
        file_paths = glob.glob(os.path.join(dir_path, file_pattern))
        for file_path in file_paths:
            os.remove(file_path)

        print('')
        print('Previous results removed in quadrant {}.'.format(quadrant))
        print('')

    ############ PLOT RAW DATA FOR QUADRANT #########
        
    plot_raw_data(df=df, quadrant=quadrant, s=.1, save_image=True)

    ############ HDBSCAN #############

    compute_hdbscan = True
    #mahalanobis, euclidean

    if compute_hdbscan:          
        clustering(df=df, min_cluster=50, min_samples=20, save_members=True, clustering_on=clustering_on, metric='euclidean', \
                   path_to_save='../data_new_clusters/shell{}/clusters{}/'.format(shell, quadrant))

    ######################

    plot_clusters = True

    if plot_clusters:
        # path to clusters found!
        path = '../data_new_clusters/shell{}/clusters{}/'.format(shell, quadrant)
        file_pattern = "*.csv"
        dir_list = glob.glob(os.path.join(path, file_pattern))
        #dir_list = os.listdir(path)

        for i in dir_list:
            #cluster = pd.read_csv(os.path.join(path, i))
            cluster = pd.read_csv(i)
            plot_results(cluster=cluster, shell=shell, quadrant=quadrant, save_fig=True)


print('')
print('Done!')
print('')

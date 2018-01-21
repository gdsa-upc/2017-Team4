from get_params import get_params
from tqdm import *
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.models import Model
from keras.applications.resnet50 import preprocess_input
import numpy as np
import os, sys
from sklearn.metrics.pairwise import pairwise_distances

"""
Baseline with deep ResNet50 features

Usage:
python deepfeats.py 1
(for batch size of 1)

Feature extraction will take about 1-2h to complete in CPU.If you have access
to a GPU, increase batch size as much as you can and it will run in no time ;)

Usage (with GPU and Theano backend):

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn,
optimizer=fast_compile python deepfeats.py 256

where 256 is the batch size. Decrease until it fits the GPU


Output:
Ranking file to be uploaded to Kaggle
"""

def extract_feats(params,bsize):

    model = ResNet50(weights='imagenet',include_top=False)

    '''
    model = Model(input=base_model.input,
                  output=base_model.get_layer('fc2').output)
    '''
    for split in ['train','val','test']:
        params['split'] = split
        dbroot = '%s/%s/%s/images'%(params['root'],
                                    params['database'],
                                    params['split'])

        savefile = '%s/%s/features/%s_resnet.npy'%(params['root'],
                                                      params['root_save'],
                                                      params['split'])

        if os.path.exists(savefile):
            continue

        imlist = '%s/%s/image_lists/%s.txt'%(params['root'],
                                             params['root_save'],
                                             params['split'])
        imlist = readfile(imlist)
        fts = []
        batch = []
        for i,im in tqdm(enumerate(imlist)):
            batch.append(read_im(os.path.join(dbroot,im.rstrip())))

            if (i+1)%bsize == 0:
                batch = np.array(batch)
                batch = np.reshape(batch,(bsize,224,224,3))
                feats = model.predict_on_batch(batch)

                if len(fts) > 0:
                    fts = np.vstack((fts,feats))
                else:
                    fts = feats
                batch = []

        if len(batch) > 0:
            batch = np.array(batch)
            batch = np.reshape(batch,(len(batch),224,224,3))
            feats = model.predict_on_batch(batch)
            fts = np.vstack((fts,feats))

        np.save(savefile,fts.squeeze())

def read_im(img_path):

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x

def readfile(f):

    with open(f,'r') as fr:
        lines = fr.readlines()
    return lines

def write_ranking_csv(params,ranking_file):

    imlist = '%s/%s/image_lists/%s.txt'%(params['root'],
                                         params['root_save'],
                                         params['split'])
    trainlist = '%s/%s/image_lists/%s.txt'%(params['root'],
                                            params['root_save'],
                                            'train')
    feats = '%s/%s/features/%s_resnet.npy'%(params['root'],
                                            params['root_save'],
                                            params['split'])
    trainfeats = '%s/%s/features/%s_resnet.npy'%(params['root'],
                                                 params['root_save'],
                                                 'train')

    imlist = readfile(imlist)
    trainlist = readfile(trainlist)

    feats = np.load(feats)
    trainfeats = np.load(trainfeats)

    print (np.shape(feats))
    dists = pairwise_distances(feats,trainfeats)


    for i,im in enumerate(imlist):

        dist = dists[i, :]
        ranking = np.array(trainlist)[np.argsort(dist)]

        ranking_file.write(im.split('.')[0] + ',')

        for item in ranking:
            ranking_file.write(item.split('.')[0] + ' ')
        ranking_file.write('\n')

    return ranking_file

def write_ranking_txt(params):

    imlist = '%s/%s/image_lists/%s.txt'%(params['root'],
                                         params['root_save'],
                                         params['split'])
    trainlist = '%s/%s/image_lists/%s.txt'%(params['root'],
                                            params['root_save'],
                                            'train')
    feats = '%s/%s/features/%s_resnet.npy'%(params['root'],
                                            params['root_save'],
                                            params['split'])
    trainfeats = '%s/%s/features/%s_resnet.npy'%(params['root'],
                                                 params['root_save'],
                                                 'train')



    imlist = readfile(imlist)
    trainlist = readfile(trainlist)

    feats = np.load(feats)
    trainfeats = np.load(trainfeats)

    print (np.shape(feats))

    for val_id in feats.keys():

        ranking= trainfeats.keys()
        X=np.array(trainfeats.values())

        dists = pairwise_distances(feats,X.squeeze())

        ranking= list(np.array(ranking)[np.argsort(dists.squeeze())])

        outfile = open(os.path.join(params['root'], params['root_save'], params['rankings_dir'], params['deepfeats'],
                                    val_id.split('.')[0] + '.txt'), 'w')


        for item in ranking:
            outfile.write(item.split('.')[0] + '\n')
        outfile.close()

def rank (params):

    imlist = '%s/%s/image_lists/%s.txt'%(params['root'],
                                         params['root_save'],
                                         params['split'])
    trainlist = '%s/%s/image_lists/%s.txt'%(params['root'],
                                            params['root_save'],
                                            'train')
    feats = '%s/%s/features/%s_resnet.npy'%(params['root'],
                                            params['root_save'],
                                            params['split'])
    trainfeats = '%s/%s/features/%s_resnet.npy'%(params['root'],
                                                 params['root_save'],
                                                 'train')

    imlist = readfile(imlist)
    trainlist = readfile(trainlist)

    feats = np.load(feats)
    trainfeats = np.load(trainfeats)

    print (np.shape(feats))
    dists = pairwise_distances(feats,trainfeats)


    for i,im in enumerate(imlist):

        dist = dists[i, :]
        ranking = np.array(trainlist)[np.argsort(dist)]

        file = open(os.path.join(params['root'], params['root_save'], params['rankings_dir'], params['descriptor_type'],
                                   params['split'], im.split('.')[0] + '.txt'), 'w')

        for item in ranking:
            file.write(item.split('.')[0] + '\n')
        file.close()


if __name__ == "__main__":

    params = get_params()

    bsize = 1

    extract_feats(params, bsize)

    ranking_file = '%s/%s/rankings/resnet_ranking.csv'%(params['root'],
                                                        params['root_save'])
    for split in ['val','test']:
        params['split'] = split
        rank(params)

    ranking_file = open(ranking_file,'w')
    ranking_file.write('Query,RetrievedDocuments\n')
    for split in ['val','test']:
        params['split'] = split
        ranking_file = write_ranking_csv(params,ranking_file)

    ranking_file.close()




#!/usr/bin/env python

# Versão modificada sem confirmação interativa

import os

import urllib.request

import tempfile

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

BASE_URL = 'http://kaldir.vc.cit.tum.de/scannet/'

RELEASE = 'v2/scans'

V1_RELEASE = 'v1/scans'

def download_file(url, out_file):

    out_dir = os.path.dirname(out_file)

    if not os.path.isdir(out_dir):

        os.makedirs(out_dir)

    if not os.path.isfile(out_file):

        print(f'  Downloading: {url}')

        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)

        f = os.fdopen(fh, 'w')

        f.close()

        urllib.request.urlretrieve(url, out_file_tmp)

        os.rename(out_file_tmp, out_file)

        print(f'  Saved: {out_file}')

    else:

        print(f'  Already exists, skipping: {out_file}')

def download_sens(scene_id, out_dir):

    # .sens usa v1 mesmo no v2

    url = BASE_URL + V1_RELEASE + '/' + scene_id + '/' + scene_id + '.sens'

    out_file = os.path.join(out_dir, scene_id, scene_id + '.sens')

    print(f'\n[{scene_id}] Downloading .sens...')

    download_file(url, out_file)

    print(f'[{scene_id}] Done.')

if __name__ == '__main__':

    scenes = [

          'scene0142_00',

          #'scene0164_00',

          #'scene0181_00',

    ]

    out_dir = 'data/scannet/scans'

    for scene in scenes:

        download_sens(scene, out_dir)

    print('\nAll downloads complete.')


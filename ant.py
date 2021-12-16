from glob import glob

ant_files = glob('data/*/Annotation_files/*.txt')

annotations = []

for f in ant_files:
    annotations.append(
        list(map(lambda n: int(n), open(f, 'r').readlines()[:2])))

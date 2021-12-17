import csv
from glob import glob
from typing import Tuple, List

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame


def to_float(l) -> List[float]:
    return list(map(lambda n: float(n), l))


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2) -> float:
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_df_keras() -> DataFrame:
    files_data = []
    csv_files = glob('out/*.csv')
    for f in csv_files:
        with open(f, newline='\n') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                l_sh_x, l_sh_y, l_sh_z = to_float(row[5:8])
                r_sh_x, r_sh_y, r_sh_z = to_float(row[8:11])
                c_sh_x = (r_sh_x + l_sh_x) / 2.0
                c_sh_y = (r_sh_y + l_sh_y) / 2.0
                c_sh_z = (r_sh_z + l_sh_z) / 2.0
                files_data.append([c_sh_x, c_sh_y, c_sh_z,
                                   1 if row[1] == 'True' else 0])

    return pd.DataFrame(np.array(files_data),
                        columns=['cx', 'cy', 'cz', 'label'])


def get_df(window=10) -> DataFrame:
    files_data = []
    csv_files = glob('out/*.csv')
    for f in csv_files:
        with open(f, newline='\n') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            d_c_sh_x = 0
            d_c_sh_y = 0
            d_c_sh_z = 0
            n_falling = 0
            for idx, row in enumerate(csv_reader):
                frame_no = int(row[0])
                falling = 1 if row[1] == 'True' else 0
                l_sh_x, l_sh_y, l_sh_z = to_float(row[5:8])
                r_sh_x, r_sh_y, r_sh_z = to_float(row[8:11])
                c_sh_x = (r_sh_x + l_sh_x) / 2.0
                c_sh_y = (r_sh_y + l_sh_y) / 2.0
                c_sh_z = (r_sh_z + l_sh_z) / 2.0
                d_c_sh_x = c_sh_x - d_c_sh_x
                d_c_sh_y = c_sh_y - d_c_sh_y
                d_c_sh_z = c_sh_z - d_c_sh_z

                if falling == 1:
                    n_falling += 1

                if idx % window == 0:
                    files_data.append(
                        [d_c_sh_x/window, d_c_sh_y/window, d_c_sh_z/window, 1 if n_falling > window / 2 else 0])
                    d_c_sh_x = 0
                    d_c_sh_y = 0
                    d_c_sh_z = 0
                    n_falling = 0

    return pd.DataFrame(np.array(files_data),
                        columns=['cx', 'cy', 'cz', 'label'])

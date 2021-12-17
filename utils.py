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
                nose_x, nose_y, nose_z = to_float(row[2:5])
                l_sh_x, l_sh_y, l_sh_z = to_float(row[5:8])
                r_sh_x, r_sh_y, r_sh_z = to_float(row[8:11])
                center_sh_x = (r_sh_x + l_sh_x) / 2.0
                center_sh_y = (r_sh_y + l_sh_y) / 2.0
                center_sh_z = (r_sh_z + l_sh_z) / 2.0
                v_c_n_x = nose_x - center_sh_x
                v_c_n_y = nose_y - center_sh_y
                v_c_n_z = nose_z - center_sh_z
                # nose_center_sh_angle = angle_between(
                #     (center_sh_x, center_sh_y, center_sh_z), (nose_x, nose_y, nose_z))
                files_data.append([nose_x, nose_y, nose_z, v_c_n_x, v_c_n_y, v_c_n_z,
                                   1 if row[1] == 'True' else 0])

    return pd.DataFrame(np.array(files_data),
                        columns=['nx', 'ny', 'nz', 'cx', 'cy', 'cz', 'label'])


def get_df(window=10) -> DataFrame:
    files_data = []
    csv_files = glob('out/*.csv')
    for f in csv_files:
        with open(f, newline='\n') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            d_nose_x = 0
            d_nose_y = 0
            d_nose_z = 0
            d_sh_x = 0
            d_sh_y = 0
            d_sh_z = 0
            d_c_sh_x = 0
            d_c_sh_y = 0
            d_c_sh_z = 0
            d_nose_center_sh_angle = 0
            d_sh_angle = 0
            n_falling = 0
            for idx, row in enumerate(csv_reader):
                frame_no = int(row[0])
                falling = 1 if row[1] == 'True' else 0
                nose_x, nose_y, nose_z = to_float(row[2:5])
                l_sh_x, l_sh_y, l_sh_z = to_float(row[5:8])
                r_sh_x, r_sh_y, r_sh_z = to_float(row[8:11])
                center_sh_x = (r_sh_x + l_sh_x) / 2.0
                center_sh_y = (r_sh_y + l_sh_y) / 2.0
                center_sh_z = (r_sh_z + l_sh_z) / 2.0
                v_c_n_x = nose_x - center_sh_x
                v_c_n_y = nose_y - center_sh_y
                v_c_n_z = nose_z - center_sh_z
                # sh_angle = angle_between(
                #     (l_sh_x, l_sh_y, l_sh_z), (r_sh_x, r_sh_y, r_sh_z))
                # nose_center_sh_angle = angle_between(
                #     (center_sh_x, center_sh_y, center_sh_z), (nose_x, nose_y, nose_z))

                d_nose_x = nose_x - d_nose_x
                d_nose_y = nose_y - d_nose_y
                d_nose_z = nose_z - d_nose_z
                d_c_sh_x = v_c_n_x - d_c_sh_x
                d_c_sh_y = v_c_n_y - d_c_sh_y
                d_c_sh_z = v_c_n_z - d_c_sh_z
                # d_nose_center_sh_angle = nose_center_sh_angle - d_nose_center_sh_angle
                # d_sh_angle = sh_angle - d_sh_angle

                if falling == 1:
                    n_falling += 1

                if idx % window == 0:
                    files_data.append(
                        [d_nose_x/window, d_nose_y/window, d_nose_z/window, d_c_sh_x/window, d_c_sh_y/window, d_c_sh_z/window, 1 if n_falling > window / 2 else 0])
                    d_nose_x = 0
                    d_nose_y = 0
                    d_nose_z = 0
                    d_sh_x = 0
                    d_sh_y = 0
                    d_sh_z = 0
                    d_c_sh_x = 0
                    d_c_sh_y = 0
                    d_c_sh_z = 0
                    n_falling = 0
                    d_nose_center_sh_angle = 0
                    d_sh_angle = 0

    return pd.DataFrame(np.array(files_data),
                        columns=['nx', 'ny', 'nz', 'cx', 'cy', 'cz', 'label'])

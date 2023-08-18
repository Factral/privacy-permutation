import sys
import os

sys.path.append(os.path.join('..', '..', 'utils'))

import utils

a = utils.Permutar(256,1)

print(a.key)
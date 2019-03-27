DEBUG = True
DEVELOPMENT = True
SECRET_KEY='mYysupersecretkeyy'
SQLALCHEMY_DATABASE_URI='mysql+pymysql://tomatouser:supersecrettomatopass@localhost/tomatoes'
SQLALCHEMY_TRACK_MODIFICATIONS = False
FILE_PATH = "/home/andrey/data/TOMATOES"

# models
TOMAT_OR_NOT_PATH = "/home/andrey/data/NN/vgg19_tomat_or_not_acc994_state_dict.md"
PLANT_HEALTH_OR_NOT_PATH = "/home/andrey/data/NN/vgg19_plant_health_or_not_acc998_state_dict.md"
TOMAT_HEALTH_OR_NOT_PATH = "/home/andrey/data/NN/vgg19_tomat_health_or_not_acc996_state_dict.md"
LEAF_OR_NOT_PATH = "/home/andrey/data/NN/vgg19_leaf-non_plant_acc990_state_dict.md"

USING_MODEL_NAME = 'vgg19'
NUM_CLASSES_USED = 2
#QUERY_AGE = 432000
QUERY_AGE = 4

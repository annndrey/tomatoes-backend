DEBUG = True
DEVELOPMENT = True
SECRET_KEY='mYysupersecretkeyy'
SQLALCHEMY_DATABASE_URI='mysql+pymysql://tomatouser:supersecrettomatopass@localhost/tomatoes'
SQLALCHEMY_TRACK_MODIFICATIONS = False
FILE_PATH = "/home/andrey/data/TOMATOES"

# models
PLANT_OR_NOT_PATH = "/home/anton/fermata/codes/a_leaf_classifier/trained_models/vgg19_plant-non_plant_acc996_060419.md"
POTATO_FIT_OR_NOT_PATH = "/home/anton/fermata/codes/a_leaf_classifier/trained_models/potat_fitoftor_or_not_no_kuzn_aug1_70-40ep.md"
#LEAF_OR_NOT_PATH = "/home/anton/fermata/codes/a_leaf_classifier/trained_models/vgg19_leaf-plant_non_leaf_acc993_060419.md"
#TOMAT_OR_NOT_PATH = "/home/anton/fermata/codes/a_leaf_classifier/trained_models/tomat_or_not_69_seed2_acc993.md"
#PLANT_HEALTH_OR_NOT_PATH = "/home/anton/fermata/codes/a_leaf_classifier/trained_models/plant_health_or_not_16_seed2_acc994.md"
#TOMAT_HEALTH_OR_NOT_PATH = "/home/anton/fermata/codes/a_leaf_classifier/trained_models/plant_health_or_not_16_seed2_acc994.md"


USING_MODEL_NAME = 'vgg19'
NUM_CLASSES_USED = 2
QUERY_AGE = 4

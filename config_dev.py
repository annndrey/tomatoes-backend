DEBUG = True
DEVELOPMENT = True
SECRET_KEY='mYysupersecretkeyy'
SQLALCHEMY_DATABASE_URI='mysql+pymysql://tomatouser:supersecrettomatopass@localhost/tomatoes'
SQLALCHEMY_TRACK_MODIFICATIONS = False
FILE_PATH = "/large_disc/PICTS/CITYFARMER"

# models
#THREE_CLASS_MODEL = "/home/anton/fermata/codes/a_leaf_classifier/trained_models/salat_health_cityfarm_50_acc1.md"
#THREE_CLASS_MODEL = "/home/anton/fermata/codes/a_leaf_classifier/trained_models/cityfarm_9classes_70_acc985.md"
#THREE_CLASS_MODEL = "/home/anton/fermata/codes/a_leaf_classifier/trained_models/cityfarm_9classes_70_20_acc988.md"
#THREE_CLASS_MODEL = "/home/anton/fermata/codes/a_leaf_classifier/trained_models/cityfarm_9classes_70_100_acc1.md"
#THREE_CLASS_MODEL = "/home/anton/fermata/codes/a_leaf_classifier/trained_models/cityfarm_9classes_70_100_20_acc1.md"
#THREE_CLASS_MODEL = "/home/anton/fermata/codes/a_leaf_classifier/trained_models/cityfarm_11classes_110_90_acc1000.md"
#THREE_CLASS_MODEL = "/home/anton/fermata/codes/a_leaf_classifier/trained_models/cityfarn_26cl_densenet169_s3_60_acc946.md"
#THREE_CLASS_MODEL = "/home/anton/fermata/codes/a_leaf_classifier/trained_models/cityfarm_30cl_110_acc969.md"
#THREE_CLASS_MODEL = "/home/anton/fermata/codes/a_leaf_classifier/trained_models/cityfarm_33cl_170_acc962.md"
THREE_CLASS_MODEL = "/home/anton/fermata/codes/a_leaf_classifier/trained_models/cityf_41cl_s3_35_acc946.md"
#THREE_CLASS_MODEL = "/home/anton/fermata/codes/a_leaf_classifier/trained_models/cityfarm_33cl_240_acc955.md"
#THREE_CLASS_MODEL = "/home/anton/fermata/codes/a_leaf_classifier/trained_models/cityf_33cl_zones_40_acc963.md"


USING_MODEL_NAME = 'densenet169'
#USING_MODEL_NAME = 'vgg19'
#NUM_CLASSES_USED = 9
#NUM_CLASSES_USED = 11
#NUM_CLASSES_USED = 33
NUM_CLASSES_USED = 41
QUERY_AGE = 4#32000

BLOCKTIME = 10
BLOCKREQUESTS = 10

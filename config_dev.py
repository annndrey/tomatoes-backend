DEBUG = True
DEVELOPMENT = True
SECRET_KEY='mYysupersecretkeyy'
SQLALCHEMY_DATABASE_URI='mysql+pymysql://tomatouser:supersecrettomatopass@localhost/tomatoes'
SQLALCHEMY_TRACK_MODIFICATIONS = False
FILE_PATH = "/home/andrey/data/TOMATOES"
#TOMAT_OR_NOT_PATH = "/media/MEDIA/NN/vgg19_deep_epochs15_batchsize20_chinese_tomat_or_not_acc9947_state_dict.md"
TOMAT_OR_NOT_PATH = "/home/andrey/data/NN/vgg19_deep_tomat_or_not_acc996_state_dict_210319.md"
PLANT_HEALTH_OR_NOT_PATH = "/home/andrey//data/NN/vgg19_deep_epochs15_batchsize20_chinese_plant_health_or_not_acc995_state_dict.md"
#TOMAT_HEALTH_OR_NOT_PATH = "/media/MEDIA/NN/vgg19_deep_epochs15_batchsize20_chinese_tomat_health_or_not_acc993_state_dict.md"
TOMAT_HEALTH_OR_NOT_PATH = "/home/andrey/data/NN/vgg19_deep_chinese_tomat_health_or_not_acc996_state_dict_210319.md"
USING_MODEL_NAME = 'vgg19'
NUM_CLASSES_USED = 2
#QUERY_AGE = 432000
QUERY_AGE = 4

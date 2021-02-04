# Download
# DATASET_DOWNLOAD_KEYWORDS = ["why are you not working"]

# for some reason the downloader is unwilling to download more than 100 per key, needs many keys
DATASET_DOWNLOAD_KEYWORDS = ["robotics competition", "indoors", "wood", "building site", "disaster interior",
                             "disaster stairs",
                             "poster", "disaster factory inside", "chernobyl inside", "wood rubble", "robocup rescue",
                             "qr code poster", "science fair", "fukushima inside", "stairs", "warehouse",
                             "hobby market",
                             "factory disaster", "inside wrecked house", "wooden construction", "corrupted image",
                             "lens flares indoors", "demolition", "wooden shack", "people", "roborave", "robocup",
                             "disaster response", "disaster response robot", "disaster response robot testing",
                             "disaster response testing facility", "rescue robot", "rubble", "inside destroyed shack",
                             "inside destroyed slums", "cardboard house", "food with labels", "supermarket", "shop",
                             "rollup",
                             "labels", "label product", "qr codes product", "robot", "boston dynamics", "electronics",
                             "pcb",
                             "industrial robot", "robot testing", "wood warehouse", "robot factory", "stair debris",
                             "indoors debris", "wood debris", "geometric shapes", "circles", "squares", "colorful dots",
                             "colourful lines", "ozobot", "robot arm", "robot tracks", "robot vision",
                             "boston dynamics testing", "lines", "inside wooden house", "camera defect",
                             "inside computer",
                             "servos", "cardboard wall", "wooden maze", "robot maze", "robot coridor",
                             "robocup rescue competition", "robocup rescue dexterity", "robocup rescue vision",
                             "roborave firefighter", "firefighter robot", "firefighter robot testing",
                             "japanese factory disaster", "derelict office", "office", "asuro robot",
                             "disaster response robot testing challenges", "indoors fire", "expo",
                             "robot testing warehouse",
                             "lego mindstorms", "robots making robots", "wooden corridor", "wooden house demolition",
                             "wooden room", "robots making house", "steampunk robots", "wood robot", "motherboard",
                             "cansat",
                             "first lego league", "line follower robot", "steam robot", "amazon warehouse",
                             "derelict warehouse", "derelict factory", "nuclear power plant disaster", "wood house",
                             "indoors mess", "burning robot", "robocup japan", "roborave firefighter",
                             "physics experiment",
                             "square labels", "circular labels", "square coloful labels", "fire symbols -label",
                             "radiation symbol -label", "burning house", "pipes", "door", "door handle", "iron bars",
                             "colourful pipes", "colourful bars", "cardboard house", "sand", "pile of sand", "gravel",
                             "pile of sand and gravel", "wooden blocks", "wooden cubes", "pile of wooden blocks",
                             "colourful wooden blocks", "tracks", "electronics", "dots", "qrcode labels",
                             "small qr codes",
                             "contours", "lego", "lego technic", "mecano", "pencisls", "plywood", "plywood wall",
                             "wooden chips", "plywood house", "electronic components", "modern art", "post modern art",
                             "cubism", "surrealism", "hand", "photo with missing pixels", "noise", "image noise",
                             "digger",
                             "toys", "rectangles", "colourful rectangles", "robocup rescue challenges",
                             "disaster response robot testing challenges", "random images", "imagenet", "outdoors",
                             "building site", "book", "map", "text", "numbers", "symbols", "small fire extinguishers",
                             "qrcode wall", "many qrcode", "heap of qr code stickers", "wooden squares", "wooden cubes",
                             "cubes", "lego", "lego blocks", "concrete blocks", "plywood blocks", "furniture",
                             "little cubes",
                             "qr stickets", "square stickers", "pile of qr codes", "pile of qr stickers", "qr kody",
                             "znicena tovarna", "havarie tovarny", "staveniste", "kostky", "dice", "wooden dice",
                             "herni kostky", "qr code dice", "qr code blocks", "qr code art", "post modern art",
                             "cubism",
                             "cubist architecture", "krychlicky", "drevene bloky", "panely", "square tiles",
                             "kitchen tiles",
                             "wooden tiles", "text", "fire", "explosive barrel -label", "wooden floor", "tiled floor",
                             "schody", "kovove tyce", "room", "concrete tiles", "dlazdice", "truhliky",
                             "cardboard boxes",
                             "people", "colourful tiles", "square pattern", "papers", "elektronika", "lepnkova stena",
                             "ctverecky", "hraci kostky", "hromada qr kodu", "stavebnice", "drevena stena",
                             "zboreny dum",
                             "prumyslovy hazard", "keyboard", "kostkovany ubrus", "dreveny nabytek"]


# Door - /m/02dgv
# Fire extinguisher - folder
# Baby doll - folder
# Person - /m/03bt1vf /m/04yx4 /m/01g317
DATASET_OBJECT_BACKGROUND_REMOVAL = {"hazmat": (1, 0),  # outer white to alpha and crop, inner white to alpha
                                     "baby_doll": (1, 1),
                                     "valve": (1, 1),
                                     "fire_extinguisher": (1, 1),
                                     "door": (1, 1),
                                     "fire_exit": (1, 0),
                                     "fire_extinguisher_sign": (1, 0),
                                     "qr_code": (0, 0)
                                     }
DATASET_OBJECT_BACKGROUND_STEP = 1
DATASET_OBJECT_ZOOM_STRENGTH = {"hazmat": 0.95,
                                "baby_doll": 0.85,
                                "valve": 0.85,
                                "fire_extinguisher": 0.7,
                                "door": 0.7,
                                "fire_exit": 0.7,
                                "fire_extinguisher_sign": 0.85,
                                "qr_code": 0.95
                                }
DATASET_OBJECT_ROTATION_STRENGTH = {"hazmat": 1.,
                                    "baby_doll": 1.,
                                    "valve": 1.,
                                    "fire_extinguisher": 1.,
                                    "door": 0.5,
                                    "fire_exit": 0.5,
                                    "fire_extinguisher_sign": 0.5,
                                    "qr_code": 1.
                                    }
DATASET_OBJECT_BLUR_CUT_STRENGTH = {"hazmat": 1.,
                                    "baby_doll": 0.8,
                                    "valve": 0.6,
                                    "fire_extinguisher": 0.9,
                                    "door": 0.8,
                                    "fire_exit": 1.,
                                    "fire_extinguisher_sign": 0.9,
                                    "qr_code": 0.8
                                    }
DATASET_OBJECT_CROP_STRENGTH = {"hazmat": 0.2,
                                "baby_doll": 0.3,
                                "valve": 0.5,
                                "fire_extinguisher": 0.5,
                                "door": 0.3,
                                "fire_exit": 0.3,
                                "fire_extinguisher_sign": 0.3,
                                "qr_code": 0.3
                                }
DATASET_OBJECT_COLOUR_FILTER_STRENGTH = {"hazmat": 0.2,
                                         "baby_doll": 0.5,
                                         "valve": 0.7,
                                         "fire_extinguisher": 0.5,
                                         "door": 1.0,
                                         "fire_exit": 0.3,
                                         "fire_extinguisher_sign": 0.3,
                                         "qr_code": 0.7
                                         }
DATASET_OBJECT_PERSPECTIVE_STRENGTH = {"hazmat": 2,
                                       "baby_doll": 0.3,
                                       "valve": 0.3,
                                       "fire_extinguisher": 0.3,
                                       "door": 0.3,
                                       "fire_exit": 0.7,
                                       "fire_extinguisher_sign": 2,
                                       "qr_code": 2
                                       }
DATASET_OBJECT_DISTORT_STRENGTH = {"hazmat": 1,
                                   "baby_doll": 0.1,
                                   "valve": 0.1,
                                   "fire_extinguisher": 0.1,
                                   "door": 0.3,
                                   "fire_exit": 0.3,
                                   "fire_extinguisher_sign": 0.4,
                                   "qr_code": 1
                                   }
DATASET_OPENIMAGES_LABEL_TO_OBJECT = {"/m/02dgv": "p/door",
                                      "/m/03bt1vf": "p/person",
                                      "/m/04yx4": "p/person",
                                      "/m/01g317": "p/person"
                                      }
DATASET_OPENIMAGES_FORBIDDEN_LABELS = ["/m/0167gd",  # doll
                                       "/m/01bl7v",  # boy
                                       "/m/0167gd"   # girl
                                       ]

DATASET_DOWNLOAD_KEYWORDS = list(set(DATASET_DOWNLOAD_KEYWORDS))
DATASET_MAX_NUM_IMAGES = 100000
DATASET_DOWNLOAD_TIME_MUL = 1
DATASET_NUM_IMAGES = DATASET_MAX_NUM_IMAGES

DATASET_OBJECT_PLACE_CHANCE = (0.03, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.07)
DATASET_FOURS_PART = 1 / 4
DATASET_MAX_OBJECTS_PER_IMG = len(DATASET_OBJECT_PLACE_CHANCE) - 1
DATASET_FOUR_MEMBER_WIGGLE = 2
DATASET_DEFAULT_SHAPE = (480, 640)
DATASET_TRAINING_SHAPES = ((96, 128), (192, 256), (288, 384), (480, 640), (480, 640))
DATASET_TRAINING_OBJECT_SIZES = (40, 70, 150, 200, 200)
DATASET_FILTERING_STRENGTHS = (0.1, 0.4, 1., 1., 0.9)

MANUAL_DATASET_SIZE = 10000
MANUAL_DATASET_FILTERING_STRENGTHS = (0.1, 0.5, 1.2, 1.5, 1.0)

DATASET_EROSION_MAX_SIZE = (3, 5, 10, 15, 10)
DATASET_SIZE_STEPS = len(DATASET_FILTERING_STRENGTHS)

# DATASET_LOCATION = "datasets/dataset_"
# VALIDATION_DATASET_LOCATION = "datasets/validation_dataset"
DATASET_LOCATION = "datasets/dataset_mixed_"
VALIDATION_DATASET_LOCATION = "datasets/validation_dataset_large"

DATASET_TRAINING_PART = 0.99
DATASET_CREATION_THREADS = 20

TRAINING_EPOCHS = ((5, 5, 0, 0), (5, 5, 10, 0), (5, 5, 5, 5), (5, 5, 5, 5), (0, 5, 5, 5))
# TRAINING_EPOCHS = ((0, 0, 0, 0), (5, 5, 10, 0), (5, 5, 5, 5), (5, 5, 5, 5), (5, 5, 5, 5))
TRAINING_LRS = ((4e-3, 4e-4, 4e-5, 4e-6), (3e-3, 3e-4, 3e-5, 3e-6), (2e-3, 2e-4, 2e-5, 2e-6), (1e-3, 1e-4, 1e-5, 1e-6), (1e-4, 1e-5, 1e-6, 1e-7))
TRAINING_REDUCE_LR_PATIENCE = 3
TRAINING_STOPPING_PATIENCE = 10
TRAINING_PATIENCE_LOSS_MARGIN = 0.3
TRAINING_RANDOM_MODIFY = False
TRAINING_LOG_PERIOD = 100
TRAINING_CHECKPOINT_PERIOD = 1800
TEST_EVALUATE = True
TEST_VISUALIZE_IMAGES = False
TEST_VISUALIZE_VIDEO = False

MAKE_DATASET = False
REDOWNLOAD_DATASET = False
REFILTER_DATASET = False
REBUILD_DATASET = False
TRAIN = False
TEST = True
TRAINING_CYCLE = 18

MODEL_LOCATION = "model_data/yolo_original.h5"
AVAILABLE_MEMORY_GB = 11
GPU_NUM = 1

DEBUG = False

VALIDATION_DATASET_SIZE = 500
VALIDATION_DATASET_FILTERING_STRENGTH = 0.3
VALIDATION_DATASET_TRAINING_SHAPE = (480, 640)

HYPERPARAMETER_SEARCH = False

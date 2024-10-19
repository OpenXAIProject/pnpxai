import os, inspect, sys
from pathlib import Path
from optimization.constants import ASSETS_DIR_NAME, RANKED_RESULTS_DIR

from utils_blended.metrics_accumulator import MetricsAccumulator
from utils_blended.video import save_video

from numpy import random
from optimization.augmentations import ImageAugmentations

from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import functional as TF
from torch.nn.functional import mse_loss
from optimization.losses import range_loss, d_clip_loss
import lpips
import numpy as np

from CLIP import clip
from guided_diffusion.guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from utils_blended.visualization import show_tensor_image, show_editied_masked_image


print(sys.path)

from configs import get_config
from utils.Evaluator import Evaluator
from utils_blended.model_normalization import ResizeWrapper

class_labels = ['tench, Tinca tinca', 'goldfish, Carassius auratus',
               'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',
               'tiger shark, Galeocerdo cuvieri', 'hammerhead, hammerhead shark',
               'electric ray, crampfish, numbfish, torpedo', 'stingray', 'cock', 'hen', 'ostrich, Struthio camelus',
               'brambling, Fringilla montifringilla', 'goldfinch, Carduelis carduelis',
               'house finch, linnet, Carpodacus mexicanus', 'junco, snowbird',
               'indigo bunting, indigo finch, indigo bird, Passerina cyanea',
               'robin, American robin, Turdus migratorius', 'bulbul', 'jay', 'magpie', 'chickadee',
               'water ouzel, dipper', 'kite', 'bald eagle, American eagle, Haliaeetus leucocephalus', 'vulture',
               'great grey owl, great gray owl, Strix nebulosa', 'European fire salamander, Salamandra salamandra',
               'common newt, Triturus vulgaris', 'eft', 'spotted salamander, Ambystoma maculatum',
               'axolotl, mud puppy, Ambystoma mexicanum', 'bullfrog, Rana catesbeiana', 'tree frog, tree-frog',
               'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
               'loggerhead, loggerhead turtle, Caretta caretta',
               'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea', 'mud turtle', 'terrapin',
               'box turtle, box tortoise', 'banded gecko', 'common iguana, iguana, Iguana iguana',
               'American chameleon, anole, Anolis carolinensis', 'whiptail, whiptail lizard', 'agama',
               'frilled lizard, Chlamydosaurus kingi', 'alligator lizard', 'Gila monster, Heloderma suspectum',
               'green lizard, Lacerta viridis', 'African chameleon, Chamaeleo chamaeleon',
               'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis',
               'African crocodile, Nile crocodile, Crocodylus niloticus',
               'American alligator, Alligator mississipiensis', 'triceratops',
               'thunder snake, worm snake, Carphophis amoenus', 'ringneck snake, ring-necked snake, ring snake',
               'hognose snake, puff adder, sand viper', 'green snake, grass snake', 'king snake, kingsnake',
               'garter snake, grass snake', 'water snake', 'vine snake', 'night snake, Hypsiglena torquata',
               'boa constrictor, Constrictor constrictor', 'rock python, rock snake, Python sebae',
               'Indian cobra, Naja naja', 'green mamba', 'sea snake',
               'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus',
               'diamondback, diamondback rattlesnake, Crotalus adamanteus',
               'sidewinder, horned rattlesnake, Crotalus cerastes', 'trilobite',
               'harvestman, daddy longlegs, Phalangium opilio', 'scorpion',
               'black and gold garden spider, Argiope aurantia', 'barn spider, Araneus cavaticus',
               'garden spider, Aranea diademata', 'black widow, Latrodectus mactans', 'tarantula',
               'wolf spider, hunting spider', 'tick', 'centipede', 'black grouse', 'ptarmigan',
               'ruffed grouse, partridge, Bonasa umbellus', 'prairie chicken, prairie grouse, prairie fowl', 'peacock',
               'quail', 'partridge', 'African grey, African gray, Psittacus erithacus', 'macaw',
               'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita', 'lorikeet', 'coucal', 'bee eater',
               'hornbill', 'hummingbird', 'jacamar', 'toucan', 'drake', 'red-breasted merganser, Mergus serrator',
               'goose', 'black swan, Cygnus atratus', 'tusker', 'echidna, spiny anteater, anteater',
               'platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus',
               'wallaby, brush kangaroo', 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus',
               'wombat', 'jellyfish', 'sea anemone, anemone', 'brain coral', 'flatworm, platyhelminth',
               'nematode, nematode worm, roundworm', 'conch', 'snail', 'slug', 'sea slug, nudibranch',
               'chiton, coat-of-mail shell, sea cradle, polyplacophore',
               'chambered nautilus, pearly nautilus, nautilus', 'Dungeness crab, Cancer magister',
               'rock crab, Cancer irroratus', 'fiddler crab',
               'king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica',
               'American lobster, Northern lobster, Maine lobster, Homarus americanus',
               'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish',
               'crayfish, crawfish, crawdad, crawdaddy', 'hermit crab', 'isopod', 'white stork, Ciconia ciconia',
               'black stork, Ciconia nigra', 'spoonbill', 'flamingo', 'little blue heron, Egretta caerulea',
               'American egret, great white heron, Egretta albus', 'bittern', 'crane', 'limpkin, Aramus pictus',
               'European gallinule, Porphyrio porphyrio',
               'American coot, marsh hen, mud hen, water hen, Fulica americana', 'bustard',
               'ruddy turnstone, Arenaria interpres', 'red-backed sandpiper, dunlin, Erolia alpina',
               'redshank, Tringa totanus', 'dowitcher', 'oystercatcher, oyster catcher', 'pelican',
               'king penguin, Aptenodytes patagonica', 'albatross, mollymawk',
               'grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus',
               'killer whale, killer, orca, grampus, sea wolf, Orcinus orca', 'dugong, Dugong dugon', 'sea lion',
               'Chihuahua', 'Japanese spaniel', 'Maltese dog, Maltese terrier, Maltese', 'Pekinese, Pekingese, Peke',
               'Shih-Tzu', 'Blenheim spaniel', 'papillon', 'toy terrier', 'Rhodesian ridgeback', 'Afghan hound, Afghan',
               'basset, basset hound', 'beagle', 'bloodhound, sleuthhound', 'bluetick', 'black-and-tan coonhound',
               'Walker hound, Walker foxhound', 'English foxhound', 'redbone', 'borzoi, Russian wolfhound',
               'Irish wolfhound', 'Italian greyhound', 'whippet', 'Ibizan hound, Ibizan Podenco',
               'Norwegian elkhound, elkhound', 'otterhound, otter hound', 'Saluki, gazelle hound',
               'Scottish deerhound, deerhound', 'Weimaraner', 'Staffordshire bullterrier, Staffordshire bull terrier',
               'American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier',
               'Bedlington terrier', 'Border terrier', 'Kerry blue terrier', 'Irish terrier', 'Norfolk terrier',
               'Norwich terrier', 'Yorkshire terrier', 'wire-haired fox terrier', 'Lakeland terrier',
               'Sealyham terrier, Sealyham', 'Airedale, Airedale terrier', 'cairn, cairn terrier', 'Australian terrier',
               'Dandie Dinmont, Dandie Dinmont terrier', 'Boston bull, Boston terrier', 'miniature schnauzer',
               'giant schnauzer', 'standard schnauzer', 'Scotch terrier, Scottish terrier, Scottie',
               'Tibetan terrier, chrysanthemum dog', 'silky terrier, Sydney silky', 'soft-coated wheaten terrier',
               'West Highland white terrier', 'Lhasa, Lhasa apso', 'flat-coated retriever', 'curly-coated retriever',
               'golden retriever', 'Labrador retriever', 'Chesapeake Bay retriever', 'German short-haired pointer',
               'vizsla, Hungarian pointer', 'English setter', 'Irish setter, red setter', 'Gordon setter',
               'Brittany spaniel', 'clumber, clumber spaniel', 'English springer, English springer spaniel',
               'Welsh springer spaniel', 'cocker spaniel, English cocker spaniel, cocker', 'Sussex spaniel',
               'Irish water spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard', 'kelpie', 'komondor',
               'Old English sheepdog, bobtail', 'Shetland sheepdog, Shetland sheep dog, Shetland', 'collie',
               'Border collie', 'Bouvier des Flandres, Bouviers des Flandres', 'Rottweiler',
               'German shepherd, German shepherd dog, German police dog, alsatian', 'Doberman, Doberman pinscher',
               'miniature pinscher', 'Greater Swiss Mountain dog', 'Bernese mountain dog', 'Appenzeller', 'EntleBucher',
               'boxer', 'bull mastiff', 'Tibetan mastiff', 'French bulldog', 'Great Dane', 'Saint Bernard, St Bernard',
               'Eskimo dog, husky', 'malamute, malemute, Alaskan malamute', 'Siberian husky',
               'dalmatian, coach dog, carriage dog', 'affenpinscher, monkey pinscher, monkey dog', 'basenji',
               'pug, pug-dog', 'Leonberg', 'Newfoundland, Newfoundland dog', 'Great Pyrenees', 'Samoyed, Samoyede',
               'Pomeranian', 'chow, chow chow', 'keeshond', 'Brabancon griffon', 'Pembroke, Pembroke Welsh corgi',
               'Cardigan, Cardigan Welsh corgi', 'toy poodle', 'miniature poodle', 'standard poodle',
               'Mexican hairless', 'timber wolf, grey wolf, gray wolf, Canis lupus',
               'white wolf, Arctic wolf, Canis lupus tundrarum', 'red wolf, maned wolf, Canis rufus, Canis niger',
               'coyote, prairie wolf, brush wolf, Canis latrans', 'dingo, warrigal, warragal, Canis dingo',
               'dhole, Cuon alpinus', 'African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus',
               'hyena, hyaena', 'red fox, Vulpes vulpes', 'kit fox, Vulpes macrotis',
               'Arctic fox, white fox, Alopex lagopus', 'grey fox, gray fox, Urocyon cinereoargenteus',
               'tabby, tabby cat', 'tiger cat', 'Persian cat', 'Siamese cat, Siamese', 'Egyptian cat',
               'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor', 'lynx, catamount',
               'leopard, Panthera pardus', 'snow leopard, ounce, Panthera uncia',
               'jaguar, panther, Panthera onca, Felis onca', 'lion, king of beasts, Panthera leo',
               'tiger, Panthera tigris', 'cheetah, chetah, Acinonyx jubatus', 'brown bear, bruin, Ursus arctos',
               'American black bear, black bear, Ursus americanus, Euarctos americanus',
               'ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus',
               'sloth bear, Melursus ursinus, Ursus ursinus', 'mongoose', 'meerkat, mierkat', 'tiger beetle',
               'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle', 'ground beetle, carabid beetle',
               'long-horned beetle, longicorn, longicorn beetle', 'leaf beetle, chrysomelid', 'dung beetle',
               'rhinoceros beetle', 'weevil', 'fly', 'bee', 'ant, emmet, pismire', 'grasshopper, hopper', 'cricket',
               'walking stick, walkingstick, stick insect', 'cockroach, roach', 'mantis, mantid', 'cicada, cicala',
               'leafhopper', 'lacewing, lacewing fly',
               "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
               'damselfly', 'admiral', 'ringlet, ringlet butterfly',
               'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus', 'cabbage butterfly',
               'sulphur butterfly, sulfur butterfly', 'lycaenid, lycaenid butterfly', 'starfish, sea star',
               'sea urchin', 'sea cucumber, holothurian', 'wood rabbit, cottontail, cottontail rabbit', 'hare',
               'Angora, Angora rabbit', 'hamster', 'porcupine, hedgehog',
               'fox squirrel, eastern fox squirrel, Sciurus niger', 'marmot', 'beaver', 'guinea pig, Cavia cobaya',
               'sorrel', 'zebra', 'hog, pig, grunter, squealer, Sus scrofa', 'wild boar, boar, Sus scrofa', 'warthog',
               'hippopotamus, hippo, river horse, Hippopotamus amphibius', 'ox',
               'water buffalo, water ox, Asiatic buffalo, Bubalus bubalis', 'bison', 'ram, tup',
               'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis',
               'ibex, Capra ibex', 'hartebeest', 'impala, Aepyceros melampus', 'gazelle',
               'Arabian camel, dromedary, Camelus dromedarius', 'llama', 'weasel', 'mink',
               'polecat, fitch, foulmart, foumart, Mustela putorius', 'black-footed ferret, ferret, Mustela nigripes',
               'otter', 'skunk, polecat, wood pussy', 'badger', 'armadillo',
               'three-toed sloth, ai, Bradypus tridactylus', 'orangutan, orang, orangutang, Pongo pygmaeus',
               'gorilla, Gorilla gorilla', 'chimpanzee, chimp, Pan troglodytes', 'gibbon, Hylobates lar',
               'siamang, Hylobates syndactylus, Symphalangus syndactylus', 'guenon, guenon monkey',
               'patas, hussar monkey, Erythrocebus patas', 'baboon', 'macaque', 'langur', 'colobus, colobus monkey',
               'proboscis monkey, Nasalis larvatus', 'marmoset', 'capuchin, ringtail, Cebus capucinus',
               'howler monkey, howler', 'titi, titi monkey', 'spider monkey, Ateles geoffroyi',
               'squirrel monkey, Saimiri sciureus', 'Madagascar cat, ring-tailed lemur, Lemur catta',
               'indri, indris, Indri indri, Indri brevicaudatus', 'Indian elephant, Elephas maximus',
               'African elephant, Loxodonta africana',
               'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens',
               'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca', 'barracouta, snoek', 'eel',
               'coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch',
               'rock beauty, Holocanthus tricolor', 'anemone fish', 'sturgeon',
               'gar, garfish, garpike, billfish, Lepisosteus osseus', 'lionfish',
               'puffer, pufferfish, blowfish, globefish', 'abacus', 'abaya',
               "academic gown, academic robe, judge's robe", 'accordion, piano accordion, squeeze box',
               'acoustic guitar', 'aircraft carrier, carrier, flattop, attack aircraft carrier', 'airliner',
               'airship, dirigible', 'altar', 'ambulance', 'amphibian, amphibious vehicle', 'analog clock',
               'apiary, bee house', 'apron',
               'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin',
               'assault rifle, assault gun', 'backpack, back pack, knapsack, packsack, rucksack, haversack',
               'bakery, bakeshop, bakehouse', 'balance beam, beam', 'balloon',
               'ballpoint, ballpoint pen, ballpen, Biro', 'Band Aid', 'banjo',
               'bannister, banister, balustrade, balusters, handrail', 'barbell', 'barber chair', 'barbershop', 'barn',
               'barometer', 'barrel, cask', 'barrow, garden cart, lawn cart, wheelbarrow', 'baseball', 'basketball',
               'bassinet', 'bassoon', 'bathing cap, swimming cap', 'bath towel', 'bathtub, bathing tub, bath, tub',
               'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon',
               'beacon, lighthouse, beacon light, pharos', 'beaker', 'bearskin, busby, shako', 'beer bottle',
               'beer glass', 'bell cote, bell cot', 'bib', 'bicycle-built-for-two, tandem bicycle, tandem',
               'bikini, two-piece', 'binder, ring-binder', 'binoculars, field glasses, opera glasses', 'birdhouse',
               'boathouse', 'bobsled, bobsleigh, bob', 'bolo tie, bolo, bola tie, bola', 'bonnet, poke bonnet',
               'bookcase', 'bookshop, bookstore, bookstall', 'bottlecap', 'bow', 'bow tie, bow-tie, bowtie',
               'brass, memorial tablet, plaque', 'brassiere, bra, bandeau',
               'breakwater, groin, groyne, mole, bulwark, seawall, jetty', 'breastplate, aegis, egis', 'broom',
               'bucket, pail', 'buckle', 'bulletproof vest', 'bullet train, bullet', 'butcher shop, meat market',
               'cab, hack, taxi, taxicab', 'caldron, cauldron', 'candle, taper, wax light', 'cannon', 'canoe',
               'can opener, tin opener', 'cardigan', 'car mirror',
               'carousel, carrousel, merry-go-round, roundabout, whirligig', "carpenter's kit, tool kit", 'carton',
               'car wheel',
               'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM',
               'cassette', 'cassette player', 'castle', 'catamaran', 'CD player', 'cello, violoncello',
               'cellular telephone, cellular phone, cellphone, cell, mobile phone', 'chain', 'chainlink fence',
               'chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour', 'chain saw, chainsaw',
               'chest', 'chiffonier, commode', 'chime, bell, gong', 'china cabinet, china closet', 'Christmas stocking',
               'church, church building', 'cinema, movie theater, movie theatre, movie house, picture palace',
               'cleaver, meat cleaver, chopper', 'cliff dwelling', 'cloak', 'clog, geta, patten, sabot',
               'cocktail shaker', 'coffee mug', 'coffeepot', 'coil, spiral, volute, whorl, helix', 'combination lock',
               'computer keyboard, keypad', 'confectionery, confectionary, candy store',
               'container ship, containership, container vessel', 'convertible', 'corkscrew, bottle screw',
               'cornet, horn, trumpet, trump', 'cowboy boot', 'cowboy hat, ten-gallon hat', 'cradle', 'crane',
               'crash helmet', 'crate', 'crib, cot', 'Crock Pot', 'croquet ball', 'crutch', 'cuirass',
               'dam, dike, dyke', 'desk', 'desktop computer', 'dial telephone, dial phone', 'diaper, nappy, napkin',
               'digital clock', 'digital watch', 'dining table, board', 'dishrag, dishcloth',
               'dishwasher, dish washer, dishwashing machine', 'disk brake, disc brake',
               'dock, dockage, docking facility', 'dogsled, dog sled, dog sleigh', 'dome', 'doormat, welcome mat',
               'drilling platform, offshore rig', 'drum, membranophone, tympan', 'drumstick', 'dumbbell', 'Dutch oven',
               'electric fan, blower', 'electric guitar', 'electric locomotive', 'entertainment center', 'envelope',
               'espresso maker', 'face powder', 'feather boa, boa', 'file, file cabinet, filing cabinet', 'fireboat',
               'fire engine, fire truck', 'fire screen, fireguard', 'flagpole, flagstaff', 'flute, transverse flute',
               'folding chair', 'football helmet', 'forklift', 'fountain', 'fountain pen', 'four-poster', 'freight car',
               'French horn, horn', 'frying pan, frypan, skillet', 'fur coat', 'garbage truck, dustcart',
               'gasmask, respirator, gas helmet', 'gas pump, gasoline pump, petrol pump, island dispenser', 'goblet',
               'go-kart', 'golf ball', 'golfcart, golf cart', 'gondola', 'gong, tam-tam', 'gown', 'grand piano, grand',
               'greenhouse, nursery, glasshouse', 'grille, radiator grille',
               'grocery store, grocery, food market, market', 'guillotine', 'hair slide', 'hair spray', 'half track',
               'hammer', 'hamper', 'hand blower, blow dryer, blow drier, hair dryer, hair drier',
               'hand-held computer, hand-held microcomputer', 'handkerchief, hankie, hanky, hankey',
               'hard disc, hard disk, fixed disk', 'harmonica, mouth organ, harp, mouth harp', 'harp',
               'harvester, reaper', 'hatchet', 'holster', 'home theater, home theatre', 'honeycomb', 'hook, claw',
               'hoopskirt, crinoline', 'horizontal bar, high bar', 'horse cart, horse-cart', 'hourglass', 'iPod',
               'iron, smoothing iron', "jack-o'-lantern", 'jean, blue jean, denim', 'jeep, landrover',
               'jersey, T-shirt, tee shirt', 'jigsaw puzzle', 'jinrikisha, ricksha, rickshaw', 'joystick', 'kimono',
               'knee pad', 'knot', 'lab coat, laboratory coat', 'ladle', 'lampshade, lamp shade',
               'laptop, laptop computer', 'lawn mower, mower', 'lens cap, lens cover',
               'letter opener, paper knife, paperknife', 'library', 'lifeboat', 'lighter, light, igniter, ignitor',
               'limousine, limo', 'liner, ocean liner', 'lipstick, lip rouge', 'Loafer', 'lotion',
               'loudspeaker, speaker, speaker unit, loudspeaker system, speaker system', "loupe, jeweler's loupe",
               'lumbermill, sawmill', 'magnetic compass', 'mailbag, postbag', 'mailbox, letter box', 'maillot',
               'maillot, tank suit', 'manhole cover', 'maraca', 'marimba, xylophone', 'mask', 'matchstick', 'maypole',
               'maze, labyrinth', 'measuring cup', 'medicine chest, medicine cabinet', 'megalith, megalithic structure',
               'microphone, mike', 'microwave, microwave oven', 'military uniform', 'milk can', 'minibus',
               'miniskirt, mini', 'minivan', 'missile', 'mitten', 'mixing bowl', 'mobile home, manufactured home',
               'Model T', 'modem', 'monastery', 'monitor', 'moped', 'mortar', 'mortarboard', 'mosque', 'mosquito net',
               'motor scooter, scooter', 'mountain bike, all-terrain bike, off-roader', 'mountain tent',
               'mouse, computer mouse', 'mousetrap', 'moving van', 'muzzle', 'nail', 'neck brace', 'necklace', 'nipple',
               'notebook, notebook computer', 'obelisk', 'oboe, hautboy, hautbois', 'ocarina, sweet potato',
               'odometer, hodometer, mileometer, milometer', 'oil filter', 'organ, pipe organ',
               'oscilloscope, scope, cathode-ray oscilloscope, CRO', 'overskirt', 'oxcart', 'oxygen mask', 'packet',
               'paddle, boat paddle', 'paddlewheel, paddle wheel', 'padlock', 'paintbrush',
               "pajama, pyjama, pj's, jammies", 'palace', 'panpipe, pandean pipe, syrinx', 'paper towel',
               'parachute, chute', 'parallel bars, bars', 'park bench', 'parking meter',
               'passenger car, coach, carriage', 'patio, terrace', 'pay-phone, pay-station',
               'pedestal, plinth, footstall', 'pencil box, pencil case', 'pencil sharpener', 'perfume, essence',
               'Petri dish', 'photocopier', 'pick, plectrum, plectron', 'pickelhaube', 'picket fence, paling',
               'pickup, pickup truck', 'pier', 'piggy bank, penny bank', 'pill bottle', 'pillow', 'ping-pong ball',
               'pinwheel', 'pirate, pirate ship', 'pitcher, ewer', "plane, carpenter's plane, woodworking plane",
               'planetarium', 'plastic bag', 'plate rack', 'plow, plough', "plunger, plumber's helper",
               'Polaroid camera, Polaroid Land camera', 'pole',
               'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria', 'poncho',
               'pool table, billiard table, snooker table', 'pop bottle, soda bottle', 'pot, flowerpot',
               "potter's wheel", 'power drill', 'prayer rug, prayer mat', 'printer', 'prison, prison house',
               'projectile, missile', 'projector', 'puck, hockey puck',
               'punching bag, punch bag, punching ball, punchball', 'purse', 'quill, quill pen',
               'quilt, comforter, comfort, puff', 'racer, race car, racing car', 'racket, racquet', 'radiator',
               'radio, wireless', 'radio telescope, radio reflector', 'rain barrel', 'recreational vehicle, RV, R.V.',
               'reel', 'reflex camera', 'refrigerator, icebox', 'remote control, remote',
               'restaurant, eating house, eating place, eatery', 'revolver, six-gun, six-shooter', 'rifle',
               'rocking chair, rocker', 'rotisserie', 'rubber eraser, rubber, pencil eraser', 'rugby ball',
               'rule, ruler', 'running shoe', 'safe', 'safety pin', 'saltshaker, salt shaker', 'sandal', 'sarong',
               'sax, saxophone', 'scabbard', 'scale, weighing machine', 'school bus', 'schooner', 'scoreboard',
               'screen, CRT screen', 'screw', 'screwdriver', 'seat belt, seatbelt', 'sewing machine', 'shield, buckler',
               'shoe shop, shoe-shop, shoe store', 'shoji', 'shopping basket', 'shopping cart', 'shovel', 'shower cap',
               'shower curtain', 'ski', 'ski mask', 'sleeping bag', 'slide rule, slipstick', 'sliding door',
               'slot, one-armed bandit', 'snorkel', 'snowmobile', 'snowplow, snowplough', 'soap dispenser',
               'soccer ball', 'sock', 'solar dish, solar collector, solar furnace', 'sombrero', 'soup bowl',
               'space bar', 'space heater', 'space shuttle', 'spatula', 'speedboat', "spider web, spider's web",
               'spindle', 'sports car, sport car', 'spotlight, spot', 'stage', 'steam locomotive', 'steel arch bridge',
               'steel drum', 'stethoscope', 'stole', 'stone wall', 'stopwatch, stop watch', 'stove', 'strainer',
               'streetcar, tram, tramcar, trolley, trolley car', 'stretcher', 'studio couch, day bed', 'stupa, tope',
               'submarine, pigboat, sub, U-boat', 'suit, suit of clothes', 'sundial', 'sunglass',
               'sunglasses, dark glasses, shades', 'sunscreen, sunblock, sun blocker', 'suspension bridge',
               'swab, swob, mop', 'sweatshirt', 'swimming trunks, bathing trunks', 'swing',
               'switch, electric switch, electrical switch', 'syringe', 'table lamp',
               'tank, army tank, armored combat vehicle, armoured combat vehicle', 'tape player', 'teapot',
               'teddy, teddy bear', 'television, television system', 'tennis ball', 'thatch, thatched roof',
               'theater curtain, theatre curtain', 'thimble', 'thresher, thrasher, threshing machine', 'throne',
               'tile roof', 'toaster', 'tobacco shop, tobacconist shop, tobacconist', 'toilet seat', 'torch',
               'totem pole', 'tow truck, tow car, wrecker', 'toyshop', 'tractor',
               'trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi', 'tray', 'trench coat',
               'tricycle, trike, velocipede', 'trimaran', 'tripod', 'triumphal arch',
               'trolleybus, trolley coach, trackless trolley', 'trombone', 'tub, vat', 'turnstile',
               'typewriter keyboard', 'umbrella', 'unicycle, monocycle', 'upright, upright piano',
               'vacuum, vacuum cleaner', 'vase', 'vault', 'velvet', 'vending machine', 'vestment', 'viaduct',
               'violin, fiddle', 'volleyball', 'waffle iron', 'wall clock', 'wallet, billfold, notecase, pocketbook',
               'wardrobe, closet, press', 'warplane, military plane',
               'washbasin, handbasin, washbowl, lavabo, wash-hand basin', 'washer, automatic washer, washing machine',
               'water bottle', 'water jug', 'water tower', 'whiskey jug', 'whistle', 'wig', 'window screen',
               'window shade', 'Windsor tie', 'wine bottle', 'wing', 'wok', 'wooden spoon', 'wool, woolen, woollen',
               'worm fence, snake fence, snake-rail fence, Virginia fence', 'wreck', 'yawl', 'yurt',
               'web site, website, internet site, site', 'comic book', 'crossword puzzle, crossword', 'street sign',
               'traffic light, traffic signal, stoplight', 'book jacket, dust cover, dust jacket, dust wrapper', 'menu',
               'plate', 'guacamole', 'consomme', 'hot pot, hotpot', 'trifle', 'ice cream, icecream',
               'ice lolly, lolly, lollipop, popsicle', 'French loaf', 'bagel, beigel', 'pretzel', 'cheeseburger',
               'hotdog, hot dog, red hot', 'mashed potato', 'head cabbage', 'broccoli', 'cauliflower',
               'zucchini, courgette', 'spaghetti squash', 'acorn squash', 'butternut squash', 'cucumber, cuke',
               'artichoke, globe artichoke', 'bell pepper', 'cardoon', 'mushroom', 'Granny Smith', 'strawberry',
               'orange', 'lemon', 'fig', 'pineapple, ananas', 'banana', 'jackfruit, jak, jack', 'custard apple',
               'pomegranate', 'hay', 'carbonara', 'chocolate sauce, chocolate syrup', 'dough', 'meat loaf, meatloaf',
               'pizza, pizza pie', 'potpie', 'burrito', 'red wine', 'espresso', 'cup', 'eggnog', 'alp', 'bubble',
               'cliff, drop, drop-off', 'coral reef', 'geyser', 'lakeside, lakeshore',
               'promontory, headland, head, foreland', 'sandbar, sand bar', 'seashore, coast, seacoast, sea-coast',
               'valley, vale', 'volcano', 'ballplayer, baseball player', 'groom, bridegroom', 'scuba diver', 'rapeseed',
               'daisy', "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum",
               'corn', 'acorn', 'hip, rose hip, rosehip', 'buckeye, horse chestnut, conker', 'coral fungus', 'agaric',
               'gyromitra', 'stinkhorn, carrion fungus', 'earthstar',
               'hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa', 'bolete',
               'ear, spike, capitulum', 'toilet tissue, toilet paper, bathroom tissue']


def L1_projection(x2, y2, eps1):
    '''
    x2: center of the L1 ball (bs x input_dim)
    y2: current perturbation (x2 + y2 is the point to be projected)
    eps1: radius of the L1 ball

    output: delta s.th. ||y2 + delta||_1 <= eps1
    and 0 <= x2 + y2 + delta <= 1
    '''

    x = x2.clone().float().view(x2.shape[0], -1)
    y = y2.clone().float().view(y2.shape[0], -1)
    sigma = y.clone().sign()
    u = torch.min(1 - x - y, x + y)
    # u = torch.min(u, epsinf - torch.clone(y).abs())
    u = torch.min(torch.zeros_like(y), u)
    l = -torch.clone(y).abs()
    d = u.clone()

    bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1)
    bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1)

    inu = 2 * (indbs < u.shape[1]).float() - 1
    size1 = inu.cumsum(dim=1)

    s1 = -u.sum(dim=1)

    c = eps1 - y.clone().abs().sum(dim=1)
    c5 = s1 + c < 0
    c2 = c5.nonzero().squeeze(1)

    s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1)

    if c2.nelement != 0:

        lb = torch.zeros_like(c2).float()
        ub = torch.ones_like(lb) * (bs.shape[1] - 1)

        # print(c2.shape, lb.shape)

        nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
        counter2 = torch.zeros_like(lb).long()
        counter = 0

        while counter < nitermax:
            counter4 = torch.floor((lb + ub) / 2.)
            counter2 = counter4.type(torch.LongTensor)

            c8 = s[c2, counter2] + c[c2] < 0
            ind3 = c8.nonzero().squeeze(1)
            ind32 = (~c8).nonzero().squeeze(1)
            # print(ind3.shape)
            if ind3.nelement != 0:
                lb[ind3] = counter4[ind3]
            if ind32.nelement != 0:
                ub[ind32] = counter4[ind32]

            # print(lb, ub)
            counter += 1

        lb2 = lb.long()
        alpha = (-s[c2, lb2] - c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]
        d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2])

    return (sigma * d).view(x2.shape)

def project_perturbation(perturbation, eps, p, center=None):
    if p in ['inf', 'linf', 'Linf']:
        pert_normalized = torch.clamp(perturbation, -eps, eps)
        return pert_normalized
    elif p in [2, 2.0, 'l2', 'L2', '2']:
        print('l2 renorm')
        pert_normalized = torch.renorm(perturbation, p=2, dim=0, maxnorm=eps)
        return pert_normalized
    elif p in [1, 1.0, 'l1', 'L1', '1']:
        ##pert_normalized = project_onto_l1_ball(perturbation, eps)
        ##return pert_normalized
        pert_normalized = L1_projection(center, perturbation, eps)
        return perturbation + pert_normalized
    #elif p in ['LPIPS']:
    #    pert_normalized = project_onto_LPIPS_ball(perturbation, eps)
    else:
        raise NotImplementedError('Projection only supports l1, l2 and inf norm')

class ImageEditor:
    def __init__(self, args) -> None:
        self.args = args
        self.probs = None
        self.y = None
        os.makedirs(self.args.output_path, exist_ok=True)

        self.ranked_results_path = Path(os.path.join(self.args.output_path, RANKED_RESULTS_DIR))
        os.makedirs(self.ranked_results_path, exist_ok=True)

        if self.args.export_assets:
            self.assets_path = Path(os.path.join(self.args.output_path, ASSETS_DIR_NAME))
            os.makedirs(self.assets_path, exist_ok=True)
        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)

        self.model_config = model_and_diffusion_defaults()
        self.model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": self.args.model_output_size == 512,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": self.args.timestep_respacing,
                "image_size": self.args.model_output_size,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 256,
                "num_head_channels": 64,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_fp16": True,
                "use_scale_shift_norm": True,
            }
        )

        # Load models
        self.device = torch.device(
            f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        print("Using device:", self.device)

        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
        self.model.num_classes = 1000
        self.model.load_state_dict(
            torch.load(
                "checkpoints/256x256_diffusion_uncond.pt"
                if self.args.model_output_size == 256
                else "checkpoints/512x512_diffusion.pt",
                map_location="cpu",
            )
        )
        self.model.requires_grad_(False).eval().to(self.device)
        for name, param in self.model.named_parameters():
            if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_()
        if self.model_config["use_fp16"]:
            self.model.convert_to_fp16()

        self.clip_model = (
            clip.load("ViT-B/16", device=self.device, jit=False)[0].eval().requires_grad_(False)
        )

        args.device = self.device
        classifier_config = get_config(args)
        # pickle.dump([args, classifier_config], open("Madry_ep3_args.pickle", "wb"))

        evaluator = Evaluator(args, classifier_config, {}, None)

        classifier = evaluator.load_model(
            3
        )

        #self.classifier = ResizeWrapper(classifier, 224)
        self.classifier = classifier
        self.classifier.to(self.device)


        self.clip_size = self.clip_model.visual.input_resolution
        self.clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)

        self.image_augmentations = ImageAugmentations(self.clip_size, self.args.aug_num)
        self.metrics_accumulator = MetricsAccumulator()

    def unscale_timestep(self, t):
        unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()

        return unscaled_timestep

    def clip_loss(self, x_in, text_embed):
        clip_loss = torch.tensor(0)

        if self.mask is not None:
            masked_input = x_in * self.mask
        else:
            masked_input = x_in
        augmented_input = self.image_augmentations(masked_input).add(1).div(2)
        clip_in = self.clip_normalize(augmented_input)
        image_embeds = self.clip_model.encode_image(clip_in).float()
        dists = d_clip_loss(image_embeds, text_embed)

        # We want to sum over the averages
        for i in range(self.args.batch_size):
            # We want to average at the "augmentations level"
            clip_loss = clip_loss + dists[i :: self.args.batch_size].mean()

        return clip_loss

    def unaugmented_clip_distance(self, x, text_embed):
        x = F.resize(x, [self.clip_size, self.clip_size])
        image_embeds = self.clip_model.encode_image(x).float()
        dists = d_clip_loss(image_embeds, text_embed)

        return dists.item()

    def edit_image_by_prompt(self):
        try:
            text_embed = self.clip_model.encode_text(
                clip.tokenize(self.args.prompt).to(self.device)
            ).float()
        except Exception as err:
            print(str(err))

        self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
        self.init_image_pil = Image.open(self.args.init_image).convert("RGB")
        self.init_image_pil = self.init_image_pil.resize(self.image_size, Image.LANCZOS)  # type: ignore
        self.init_image = (
            TF.to_tensor(self.init_image_pil).to(self.device).unsqueeze(0).mul(2).sub(1)
        )

        if self.args.export_assets:
            img_path = self.assets_path / Path(self.args.output_file)
            self.init_image_pil.save(img_path)

        self.mask = torch.ones_like(self.init_image, device=self.device)
        self.mask_pil = None
        if self.args.mask is not None:
            self.mask_pil = Image.open(self.args.mask).convert("RGB")
            if self.mask_pil.size != self.image_size:
                self.mask_pil = self.mask_pil.resize(self.image_size, Image.NEAREST)  # type: ignore
            image_mask_pil_binarized = ((np.array(self.mask_pil) > 0.5) * 255).astype(np.uint8)
            if self.args.invert_mask:
                image_mask_pil_binarized = 255 - image_mask_pil_binarized
                self.mask_pil = TF.to_pil_image(image_mask_pil_binarized)
            self.mask = TF.to_tensor(Image.fromarray(image_mask_pil_binarized))
            self.mask = self.mask[0, ...].unsqueeze(0).unsqueeze(0).to(self.device)

            if self.args.export_assets:
                mask_path = self.assets_path / Path(
                    self.args.output_file.replace(".png", "_mask.png")
                )
                self.mask_pil.save(mask_path)


        def cond_fn(x, t, y=None):
            if self.args.prompt == "":
                return torch.zeros_like(x)

            with torch.enable_grad():
                x = x.detach().requires_grad_()
                t = self.unscale_timestep(t)

                out = self.diffusion.p_mean_variance(
                    self.model, x, t, clip_denoised=False, model_kwargs={"y": y}
                )

                fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
                x_in = out["pred_xstart"] * fac + x * (1 - fac)
                # x_in = out["pred_xstart"]

                loss = torch.tensor(0)
                if self.args.clip_guidance_lambda != 0:
                    print('using clip guidance')
                    clip_loss = self.clip_loss(x_in, text_embed) * self.args.clip_guidance_lambda
                    loss = loss + clip_loss
                    self.metrics_accumulator.update_metric("clip_loss", clip_loss.item())
                if self.args.classifier_lambda != 0:
                    loss_temp = torch.tensor(0.0).to(self.device)
                    if self.mask is not None:
                        masked_input = x_in * self.mask
                    else:
                        masked_input = x_in

                    logits = self.classifier(self.image_augmentations(masked_input).add(1).div(2))
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    #loss_indiv = log_probs[range(len(logits)), y.view(-1)]
                    loss_indiv = log_probs[range(self.args.batch_size*self.args.aug_num), y.view(-1).repeat(self.args.aug_num)]
                    for i in range(self.args.batch_size):
                        # We want to average at the "augmentations level"
                        loss_temp += loss_indiv[i:: self.args.batch_size].mean()
                    print('shape loss', loss_indiv.shape)
                    print('targets', y.shape, y)
                    print('probs', logits[:self.args.batch_size].softmax(1)[range(self.args.batch_size), y.view(-1)])
                    print('dist', (self.init_image-x_in).view(len(logits), -1).norm(p=2, dim=1))
                    self.probs = logits[:self.args.batch_size].softmax(1)[range(self.args.batch_size), y.view(-1)]
                    self.y = y
                    classifier_loss = loss_temp * self.args.classifier_lambda
                    loss = loss - classifier_loss
                    self.metrics_accumulator.update_metric("classifier_loss", classifier_loss.item())

                if self.args.range_lambda != 0:
                    r_loss = range_loss(out["pred_xstart"]).sum() * self.args.range_lambda
                    loss = loss + r_loss
                    self.metrics_accumulator.update_metric("range_loss", r_loss.item())

                if self.args.background_preservation_loss:
                    if self.mask is not None:
                        print('using mask')
                        masked_background = x_in * (1 - self.mask)
                        ##masked_background = x_in * self.mask
                    else:
                        print('not using mask')
                        masked_background = x_in

                    if self.args.lpips_sim_lambda:
                        loss = (
                            loss
                            + self.lpips_model(masked_background, self.init_image).sum()
                            * self.args.lpips_sim_lambda
                        )
                    if self.args.l2_sim_lambda:
                        print('using l2 sim', self.args.l2_sim_lambda)
                        loss = (
                            loss
                            #+ ((masked_background - self.init_image).view(len(self.init_image), -1).norm(p=1.5, dim=1)**1.5).mean() * self.args.l2_sim_lambda
                            + mse_loss(masked_background, self.init_image) * self.args.l2_sim_lambda
                        )
                        #print('1.5 scaled loss', ((masked_background - self.init_image).view(len(self.init_image), -1).norm(p=1.5, dim=1)**1.5).mean() * self.args.l2_sim_lambda)
                        print('mse scaled loss', mse_loss(masked_background, self.init_image) * self.args.l2_sim_lambda)
                        print('total losss', loss)

                return -torch.autograd.grad(loss, x)[0]

        @torch.no_grad()
        def postprocess_fn(out, t):
            if self.mask is not None:
                print('postprocessing mask')
                background_stage_t = self.diffusion.q_sample(self.init_image, t[0])
                background_stage_t = torch.tile(
                    background_stage_t, dims=(self.args.batch_size, 1, 1, 1)
                )
                out["sample"] = out["sample"] * self.mask + background_stage_t * (1 - self.mask)
                # try l2 projection
                #eps = 1500 / 75
                #print('projecting', max(eps*t[0], 160), t[0])
                #out["sample"] = background_stage_t + project_perturbation(out["sample"] * self.mask - background_stage_t * self.mask, eps=max(eps*t[0], 160), p=2)
            return out

        save_image_interval = self.diffusion.num_timesteps // 5
        targets_classifier = [self.args.target_class]*4 #[293]*4 #[979]*4 #[293, 293, 293, 293] #[286, 287, 293, 288] #[979, 973, 980, 974] [286, 287, 293, 288]
        for iteration_number in range(self.args.iterations_num):
            # Here iterate over the dataloader of ImageNet-S

            print(f"Start iteration {iteration_number}")

            samples = self.diffusion.p_sample_loop_progressive(
                self.model,
                (
                    self.args.batch_size,
                    3,
                    self.model_config["image_size"],
                    self.model_config["image_size"],
                ),
                clip_denoised=False,
                model_kwargs={}
                if self.args.model_output_size == 256 and self.args.classifier_lambda == 0
                else {

                    "y": torch.tensor(targets_classifier, device=self.device, dtype=torch.long)#torch.zeros([self.args.batch_size], device=self.device, dtype=torch.long)
                },
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=self.args.skip_timesteps,
                init_image=self.init_image,
                postprocess_fn=None if self.args.local_clip_guided_diffusion else postprocess_fn,
                randomize_class=False,
            )

            intermediate_samples = [[] for i in range(self.args.batch_size)]
            total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1
            for j, sample in enumerate(samples):
                should_save_image = j % save_image_interval == 0 or j == total_steps
                if should_save_image or self.args.save_video:
                    self.metrics_accumulator.print_average_metric()

                    for b in range(self.args.batch_size):
                        pred_image = sample["pred_xstart"][b]
                        visualization_path = Path(
                            os.path.join(self.args.output_path, self.args.output_file)
                        )
                        visualization_path = visualization_path.with_name(
                            f"{visualization_path.stem}_i_{iteration_number}_b_{b}{visualization_path.suffix}"
                        )

                        if (
                            self.mask is not None
                            and self.args.enforce_background
                            and j == total_steps
                            and not self.args.local_clip_guided_diffusion
                        ):
                            pred_image = (
                                self.init_image[0] * (1 - self.mask[0]) + pred_image * self.mask[0]
                            )
                        pred_image = pred_image.add(1).div(2).clamp(0, 1)
                        pred_image_pil = TF.to_pil_image(pred_image)
                        masked_pred_image = self.mask * pred_image.unsqueeze(0)

                        final_distance = self.unaugmented_clip_distance(
                            masked_pred_image, text_embed
                        )

                        #final_distance = self.probs[b]
                        formatted_distance = f"{final_distance:.4f}"

                        if self.args.export_assets:
                            pred_path = self.assets_path / visualization_path.name
                            pred_image_pil.save(pred_path)

                        if j == total_steps:
                           path_friendly_distance = formatted_distance.replace(".", "")

                           ranked_pred_path = self.ranked_results_path / (
                               #str(self.args.classifier_lambda) + "_" + str(self.args.l2_sim_lambda) + '_classifier_' + path_friendly_distance + "_" + class_labels[self.y[b].item()] + visualization_path.name
                               path_friendly_distance + "_" + visualization_path.name
                            )
                           pred_image_pil.save(ranked_pred_path)

                        intermediate_samples[b].append(pred_image_pil)
                        if should_save_image:
                            show_editied_masked_image(
                                title=self.args.prompt,
                                source_image=self.init_image_pil,
                                edited_image=pred_image_pil,
                                mask=self.mask_pil,
                                path=visualization_path,
                                distance=formatted_distance,
                            )

            if self.args.save_video:
                for b in range(self.args.batch_size):
                    video_name = self.args.output_file.replace(
                        ".png", f"_i_{iteration_number}_b_{b}.avi"
                    )
                    video_path = os.path.join(self.args.output_path, video_name)
                    save_video(intermediate_samples[b], video_path)

    def reconstruct_image(self):
        init = Image.open(self.args.init_image).convert("RGB")
        init = init.resize(
            self.image_size,  # type: ignore
            Image.LANCZOS,
        )
        init = TF.to_tensor(init).to(self.device).unsqueeze(0).mul(2).sub(1)
        targets_classifier = [self.args.target_class]*4 #[293]*4 #[979]*4 #[293, 293, 293, 293] #[286, 287, 293, 288] #[979, 973, 980, 974] #[286, 287, 293, 288]
        samples = self.diffusion.p_sample_loop_progressive(
            self.model,
            (1, 3, self.model_config["image_size"], self.model_config["image_size"],),
            clip_denoised=False,
            model_kwargs={}
            if self.args.model_output_size == 256 and self.args.classifier_lambda == 0
            else {"y": torch.tenosr(targets_classifier, device=self.device, dtype=torch.long)}, #{"y": torch.zeros([self.args.batch_size], device=self.device, dtype=torch.long)},
            cond_fn=None,
            progress=True,
            skip_timesteps=self.args.skip_timesteps,
            init_image=init,
            randomize_class=False,
        )
        save_image_interval = self.diffusion.num_timesteps // 5
        max_iterations = self.diffusion.num_timesteps - self.args.skip_timesteps - 1

        for j, sample in enumerate(samples):
            if j % save_image_interval == 0 or j == max_iterations:
                print()
                filename = os.path.join(self.args.output_path, self.args.output_file)
                TF.to_pil_image(sample["pred_xstart"][0].add(1).div(2).clamp(0, 1)).save(filename)

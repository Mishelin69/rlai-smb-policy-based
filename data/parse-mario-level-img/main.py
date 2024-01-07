from pathlib import Path
import sys
import os
import numpy as np
import json

from PIL import Image
from numpy.typing import NDArray

from typing import List, Tuple


ImageData = NDArray
def load_matches() -> List[Tuple[ImageData, float]]:

    matches: List[Tuple[ImageData, float]] = []

    if not os.path.isfile("./match.json"):

        print(f"Couldn't find \"match.json\". Make sure the file is correctly place inside this directory! {__file__}")
        exit(-1)

    with open("./match.json", "r") as f:

        dat = json.load(f)

        for obj in dat["matches"]:

            img = np.asarray(Image.open(obj["path"]).convert("L"))
            matches.append((img, float(obj["value"])))

    return matches

def main() -> None:
    
    if len(sys.argv) < 2:

        print("Expected at least input directory as parameter and got nothing!")
        exit(-1)

    #absolute path just in case
    path: str = os.path.abspath(sys.argv[1])

    if not os.path.isdir(path):

        print(f"Specified directory does not exist! ({path})")
        exit(-1)

    #check for tile matches
    #for now empty because I'm lazy but I'll make it work

    matches: List[Tuple[ImageData, float]] = load_matches()

    if not os.path.isdir("./out/"):
        print("Output directory not present creating one!")
        os.mkdir("./out")

    for img_path in os.listdir(path):
        
        obj = parse_image(os.path.abspath(path + "/" + img_path), matches)

        with open(f"./out/{str(obj[1][0])}x{str(obj[1][1])}_{img_path}.dat", "w") as f:

            f.write(obj[0])

def is_equal(a: NDArray, a_xoff: int, a_yoff: int, b: NDArray) -> bool:

    
    #
    #NOTE: This when uncommented can be used to get tiles that are not of the same colour 
    #      Good when you want to save tiles so they match over here
    #
    # arr = []

    # for i in range(16):
    #     for j in range(16):

    #         arr.append(a[a_yoff*16 + i][a_xoff*16 + j])

    # if len(set(arr)) == 1:
    #     return False


    # _arr = np.reshape(np.array(arr), (16, 16))
    # Image.fromarray(_arr).show()
    # Image.fromarray(_arr).save("./sprite/brick2.png")

    # input()

    for i in range(16):
        for j in range(16):

            if a[a_yoff*16 + i][a_xoff*16 + j] != b[i][j]:
                return False

    return True

def parse_image(path_to_img: str, matches: List[Tuple[ImageData, float]]) -> Tuple[str, Tuple[int, int]]:

    img: NDArray = np.asarray(Image.open(path_to_img).convert("L"))

    img_size = img.shape

    if img_size[0] % 16 != 0 or img_size[1] % 16 != 0:
        print("Could not split image into 16x16 tiles, please make it so it's possible!")
        exit(-1)

    y_tiles: int = int(img_size[0] / 16)
    x_tiles: int = int(img_size[1] / 16)

    out_buffer: str = ""
    print(img_size)

    for y in range(y_tiles):
        for x in range(x_tiles):

            check: bool = False

            #compare tiles to see if any matches ...
            for m in matches:

                if is_equal(img, x, y, m[0]):

                    out_buffer += str(m[1]) + " "
                    check = True
                    break

            if not check:
                out_buffer += "0 "


    return (out_buffer, (y_tiles, x_tiles))

if __name__ == '__main__':
    main()

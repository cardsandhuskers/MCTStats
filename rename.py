import os
import pandas as pd

def search_and_replace(oldString, newString):
    userDir = os.listdir(path='plugins')

    for y in userDir:
        if ("." in y):
            pass
        else:
            filename = "plugins/" + y + "/"
            userFiles = os.listdir(path=filename)

            for x in userFiles:
                if ".csv" in x:
                    fileName = "plugins/" + y + "/" + x
                    print(fileName)
                    df = pd.read_csv(fileName)
                    cols = []

                    if 'Player' in df:
                        cols += ['Player']
                    if 'Name' in df:
                        cols += ['Name']
                    if 'deadName' in df:
                        cols += ['deadName']
                    if 'killerName' in df:
                        cols += ['killerName']
                    if 'killer' in df:
                        cols += ['killer']
                    if 'prey' in df:
                        cols += ['prey']
                    if 'winningPlayer' in df:
                        cols += ['winningPlayer']
                    if 'HunterName' in df:
                        cols += ['HunterName']

                    print(cols)

                    for col_name in cols:
                        df[col_name] = df[col_name].astype(str)
                        df[col_name] = df[col_name].str.replace(oldString, newString, regex=True)

                    df.to_csv(fileName, index=False)

            # Remove non-CSV files after processing
            '''for file in userFiles:
                if not file.endswith(".csv"):
                    file_path = os.path.join(filename, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)'''


if __name__ == '__main__':
    # current: [past]
    names = {
        "maxsm_": ["ehzmax"],
        "candleflint": ["ragdoll554"],
        "j_scotty_": ["solomon_az", "j_scotty__"],
        "R41NY__": ["Cxxdii"],
        "TreeGameDev": ["treeiy", "GreenPeas420"],
        "earthside": ["Earthside", "kittycatearth"]
    }

    for new_name, old_names in names.items():
        for old_name in old_names:
            search_and_replace(old_name, new_name)

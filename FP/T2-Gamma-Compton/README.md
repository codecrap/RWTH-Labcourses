@author: Alex

I made a copy of PraktLib.py inside the T2 folder, because the python scripts stopped finding it while trying to import it. That problem came out of nowhere and I have no idea how to fix it, so I made this local copy as a temporary solution. 

@author: Olex

You have to

    import sys

and append the path of the files you want to include

    sys.path.append("./../../")

see activity.py for ex.

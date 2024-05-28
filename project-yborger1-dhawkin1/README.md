In order to run the code, the first thing that must occur is that the individual must be on the Swarthmore CS system, opened in this particular project directory. 
Once on the Swarthmore CS system, the individual should type out 'source /usr/swat/bin/CS63env' and then 'jupyter lab.' 
From there, the individual can run each cell of the Jupyter Notebook and the code should compile.

As a note, we ran into an issue ourselves, where the dataset worked but errors came up, so we found a solution to it.
If there is an issue with opening the files under the cell with the oneFlower method, there may be a need to download the images from the Kaggle. 
In this case, to make sure the rest of the program runs properly, download the dataset from this link: https://www.kaggle.com/datasets/alxmamaev/flowers-recognition 
Then, move the downloaded zip folder (normally the original downloaded folder will be labeled archive.zip) to your specific Scratch. 
After that, create a folder in your Scratch area and call it 'flowers' (and while this may seem redundant, it does work for our specific code).
Following the folder creation, right click on the zipped folder you had just downloaded and extract all of that information into the newly created 'flowers' folder. 
Finally, go into the Jupyter Notebook and find the line " '/scratch/dhawkin1/flowers/flowers/' + name " in the oneFlower method, and replace 'dhawkin1' with your own csmajor1 Swarthmore identification.

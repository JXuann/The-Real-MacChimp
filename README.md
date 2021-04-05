# The Real MacChimp?

Reliably obtaining the identifies of wild animals is an inseparable step of understanding the wild animals and their interactions, as well as monitoring the populations for conservation. Automatically recognizing wild chimpanzee (Pan troglodytes) from still images or videos faces challenges like limited labeled data, significant class imbalance, and a wide range of noises in training and testing. 

In this 'The Real MacChimp?' project, we investigate the feasibility of using (transductive) transfer learning and semi-supervised Generative Adversarial Networks to utilize both the limited labeled and unlabeled still images of chimpanzee faces to predict their identities. As a prototype, we aim to develop of an open source framework that later will be able predict multiple attributes of chimpanzees such as identity, age, age group and gender, which could be also adapted to other wild species.

Use SGAN_train.py to launch training of SGAN (adapting the paths to yours and defining the training parameters).\

The report of our experiments could be accessed via https://www.overleaf.com/read/rbxyxqhrbtgd. We recommend to use the report as an orientation to our implementation.\

Last but not the least, many thanks to Freytag and colleagues for initially assembling the dataset and make them publically available since then. The main dataset used in our work could be found here https://github.com/cvjena/chimpanzee\_faces.\

^ MacChimp stands for 'Machine-created Chimp' \
** The real McCoy: https://en.wikipedia.org/wiki/The_real_McCoy

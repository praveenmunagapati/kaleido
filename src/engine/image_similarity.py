#!/usr/bin/env python
# -*- coding: utf-8 -*-


# coding: utf-8

# Similar Image Search Using Deep Learning on Caltech-101
# =======================================================

# In this notebook, we will walk you through adapting a neural network trained on the ImageNet Challenge to find similar images within the Caltech-101 dataset. At the end of the notebook, we will be able to alogithmically identify images that are visually similar to each other within the Caltech-101 dataset.
# 
# The notebook has several parts:
# 
# * Part I focuses on loading the data.
# 
# * Part II focuses on using a pre-trained neural net to extract visual features. 
# 
# * Part III focuses on using the extracted visual features to train a nearest neighbors model. 

# Part I: The Data
# =========================================
# 
# In this notebook, we use the Caltech-101 dataset. Caltech-101 contains photos of objects belonging to 101 categories. **Note: This is a large dataset, so it may take a while to download.**It was collected by Fei-Fei Li, Marco Andreetto, and Marc 'Aurelio Ranzato in September 2003.
# 

# In[1]:

import graphlab 
images = graphlab.SFrame('http://s3.amazonaws.com/dato-datasets/caltech_101/caltech_101_images')


# Part II: Extracting Features 
# =========================================
# 
# We use the neural network trained on the 1.2 million images of the ImageNet Challenge. For each image of the Caltech-101 dataset, we take the activations of the layer before the classification layer and consider that our feature vector for the image. This is a sort of internal representation of what the network knows about the image. If these feature vectors are similar for two images, then the images should be similar as well. This concept is covered more thoroughly in our [blog post](http://blog.graphlab.com/deep-learning-blog-post) on the subject. Note that feature exctraction will not be feasible without a GPU and the [GPU installation](http://graphlab.com/products/create/gpu_install.html). **In that case, you should download the SArray that contains the result of this step.** 

# In[2]:

# Only do this if you have a GPU
#pretrained_model = graphlab.load_model('http://s3.amazonaws.com/dato-datasets/deeplearning/imagenet_model_iter45')
#images['extracted_features'] = pretrained_model.extract_features(images)

# If you do not have a GPU, do this instead. 
images['extracted_features'] = graphlab.SArray('http://s3.amazonaws.com/dato-datasets/deeplearning/pre_extracted_features.gl')


# **Now, let's inspect the images SFrame. The 'extracted_features' column contains  vector representations of the data, as we expected it to. **

# In[3]:

images


# Part III: Finding similar images via Nearest Neighbors on Extracted Features
# =========================================
# 
# Knowing that similar extracted features should mean visually similar images, we can do a similar image search simply by finding an images nearest neighbors in the feature space. We demonstrate this below. 

# **First, we construct the nearest neighbors model on the extracted features. This will allow us to see each image's closest neighbor**

# In[4]:

nearest_neighbor_model = graphlab.nearest_neighbors.create(images, features=['extracted_features'])


# In[5]:

similar_images = nearest_neighbor_model.query(images, k = 2)


# **similar_images is an SFrame which contains a query label, and it's neighbor, the reference label**

# In[6]:

similar_images


# **We do some cleaning to remove the instances where the query equals the reference. This happened beacause the query set was identical to the reference set**

# In[7]:

similar_images = similar_images[similar_images['query_label'] != similar_images['reference_label']]


# In[8]:

similar_images


# **Now we can explore similar images. For instance, the closest image to image 9 is image 1710. We can view and see both are starfish**

# In[9]:

graphlab.canvas.set_target('ipynb')
graphlab.SArray([images['image'][9]]).show()


# In[10]:

graphlab.SArray([images['image'][1710]]).show()


# **Similarly, images 0 and 1535 are two similar photos of the same person**

# In[11]:

graphlab.SArray([images['image'][0]]).show()


# In[12]:

graphlab.SArray([images['image'][1535]]).show()


from django.db import models

from django.contrib import admin
from django.contrib.auth.models import User

class Tag(models.Model):
    tag = models.CharField(max_length=50)

class Image(models.Model):
    

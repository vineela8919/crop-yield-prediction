from django.db import models

# Create your models here.

class storedatamodel(models.Model):

    state = models.CharField(max_length=500)
    dist = models.CharField(max_length=300)
    yeild = models.CharField(max_length=300)
    year = models.CharField(max_length=300)
    label = models.CharField(max_length=255)


    def __str__(self):
        return self.state,self.dist,self.yeild,self.year,self.label
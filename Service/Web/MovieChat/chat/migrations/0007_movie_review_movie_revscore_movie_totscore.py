# Generated by Django 4.2.3 on 2023-08-25 07:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0006_movie_poster'),
    ]

    operations = [
        migrations.AddField(
            model_name='movie',
            name='review',
            field=models.CharField(max_length=1000, null=True),
        ),
        migrations.AddField(
            model_name='movie',
            name='revscore',
            field=models.CharField(max_length=100, null=True),
        ),
        migrations.AddField(
            model_name='movie',
            name='totscore',
            field=models.CharField(max_length=100, null=True),
        ),
    ]

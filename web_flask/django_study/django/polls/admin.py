from django.contrib import admin

# Register your models here.

from polls.models import Question, Choise

admin.site.register(Question)
admin.site.register(Choise)
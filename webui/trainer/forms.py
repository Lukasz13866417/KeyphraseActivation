from django import forms


class TrainingRequestForm(forms.Form):
    key_phrase = forms.CharField(label="Keyphrase", max_length=255)
    num_positives = forms.IntegerField(label="Positives", min_value=10, initial=100)
    num_confusers = forms.IntegerField(label="Confusers", min_value=10, initial=100)
    num_inbetween = forms.IntegerField(label="In-between Negatives", min_value=10, initial=150)
    num_plain_negatives = forms.IntegerField(label="Plain Negatives", min_value=10, initial=100)
    growth_constant = forms.IntegerField(label="Growth Constant", min_value=0, initial=5)


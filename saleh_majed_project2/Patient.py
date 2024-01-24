class Patient:
    def __init__(self, age, gender, bmi, region, num_children, insurance_charges, smoker):
        self._age = age
        self._gender = gender
        self._bmi = bmi
        self._region = region
        self._num_children = num_children
        self._insurance_charges = insurance_charges
        self._smoker = smoker

    # Getter and Setter methods for each attribute
    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, value):
        self._age = value

    @property
    def gender(self):
        return self._gender

    @gender.setter
    def gender(self, value):
        self._gender = value

    @property
    def bmi(self):
        return self._bmi

    @bmi.setter
    def bmi(self, value):
        self._bmi = value

    @property
    def region(self):
        return self._region

    @region.setter
    def region(self, value):
        self._region = value

    @property
    def num_children(self):
        return self._num_children

    @num_children.setter
    def num_children(self, value):
        self._num_children = value

    @property
    def insurance_charges(self):
        return self._insurance_charges

    @insurance_charges.setter
    def insurance_charges(self, value):
        self._insurance_charges = value

    @property
    def smoker(self):
        return self._smoker

    @smoker.setter
    def smoker(self, value):
        self._smoker = value

    def __str__(self):
        return f"Patient(age={self.age}, gender={self.gender}, bmi={self.bmi}, region={self.region}, " \
               f"num_children={self.num_children}, insurance_charges={self.insurance_charges}, smoker={self.smoker})"

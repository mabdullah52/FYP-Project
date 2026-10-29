"""Brock University (PanCan) pulmonary nodule malignancy model - full model with spiculation
(McWilliams et al., N Engl J Med 2013;369:910-919, model 2b).

Clinical branch of the system: uses the patient's lifestyle / clinical data (age, sex, family history
of lung cancer, emphysema) plus nodule features (size, type, upper-lobe location, count, spiculation).
It is shown NEXT TO the image-based AI cancer risk, not merged with it (the two were not jointly
validated on our data). Validated for nodules found in screening of current/former smokers.
"""
import math

INTERCEPT = -6.7892
NODULE_TYPE = {"solid": 0.0, "part-solid": 0.377, "nonsolid": -0.1276}


def brock_risk(age, female, family_history, emphysema, size_mm, nodule_type="solid",
               upper_lobe=False, nodule_count=1, spiculated=False):
    """Probability (0-1) that the nodule is malignant. Returns None if age or size are missing."""
    if age is None or size_mm is None or size_mm <= 0:
        return None
    z = (INTERCEPT
         + 0.0287 * (float(age) - 62)
         + 0.6011 * bool(female)
         + 0.2961 * bool(family_history)
         + 0.2953 * bool(emphysema)
         - 5.3854 * ((float(size_mm) / 10.0) ** -0.5 - 1.58113883)
         + NODULE_TYPE.get(nodule_type or "solid", 0.0)
         + 0.6581 * bool(upper_lobe)
         - 0.0824 * (max(1, int(nodule_count)) - 4)
         + 0.7729 * bool(spiculated))
    return 1.0 / (1.0 + math.exp(-z))


def band(p):
    if p is None:
        return None
    return "low" if p < 0.05 else ("intermediate" if p < 0.30 else "high")


if __name__ == "__main__":
    # sanity: a 62-year-old man, solid 4 mm nodule -> very low; a 70-year-old woman with emphysema and a
    # spiculated 25 mm upper-lobe nodule -> high
    print(round(brock_risk(62, False, False, False, 4), 4), round(brock_risk(70, True, True, True, 25, "solid", True, 1, True), 3))

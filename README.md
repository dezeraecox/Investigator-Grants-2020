# Behind the scenes - Investigators2020

This repository contains source code for analysis of the NHMRC Fellowship scheme outcomes 2015 - 2020. This work also supported the analysis described in "[Emerging researchers face uphill struggle at NHMRC](https://www.researchprofessionalnews.com/rr-funding-insight-2020-9-emerging-researchers-face-uphill-struggle-at-nhmrc/)", originally published on September 18th 2020.

## Raw data

The raw data used in this analysis came from three main sources.

The first was the NHMRC grant outcomes website, which offers spreadsheet summaries for grants released since 2013. I chose to use only the data from 2015 onwards, for a couple of reasons: (1) the structure of the Fellowship system appears to have changed in 2014 to the ECF, CDF, RF layout which remained in place until 2018. This meant that 2013 data was poorly correlated with the more recent datasets. (2) The 2014 dataset did not have as much detail in the gender, age, state breakdowns that could be easily compared to the following years. (3) Five years seemed like a nice time period to work with!

The second source of data was the Field of Research codes used to classify research. You can find the complete list at the Australian Bureau of Statistics.

The last source of data was Scival, which I used to gather the number of research publications and average Field-Weighted-Citation-Impact (FWCI) for each awardee in the ten years previous to their year of award. This was somewhat of a manual process, and I used the 'best match' profile for each awardee imported into SciVal. Overall, 88% of the awardees were matched accurately (and this could be increased with a little manual curation). If you're interested in this process, feel free to get in touch.

## Disclaimer

This analysis was intended to inform my personal decision of whether to apply for an Investigator Grant in the upcoming rounds. It is not intended as a complete, in-depth assessment of the outcomes. Therefore, the information contained here and in the original post is provided on an “as is” basis with no guarantees of completeness, accuracy, usefulness or timeliness. Any action you take as a result of this information is done so at your own peril. If you do decide to act on this information, I wish you the best of luck whichever path you may choose. May the odds be ever in your favour.

This being said, I have of course aimed to be as unbiased and informative as possible. I have limited experience in data-science-for-public-consumption, so if you do notice any overt errors or bugs feel free to raise an issue and I will check it out as soon as possible.

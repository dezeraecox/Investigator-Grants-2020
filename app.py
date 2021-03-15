import os, re
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import altair_catplot as altcat
import geopandas
from PIL import Image
import json
import seaborn as sns

# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():

    # Set a pp title and icon for browser tab
    favicon = Image.open('utilities/icon.png')
    st.set_page_config(page_title='Investigators', page_icon = favicon, layout = 'wide', initial_sidebar_state = 'auto')

    st.markdown(
        """
    <style>
    canvas {
        max-width: 100%!important;
        height: auto!important;
    }
        </style>
        """, unsafe_allow_html=True
    )

    # st.title("NHMRC Fellowship Funding Outcomes 2015 - 2020")
    # # Once we have the dependencies, add a selector for the app mode on the sidebar.

    logo_box = st.sidebar.empty()
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Which dataset would you like to explore?",
        [" ", "Research Area", "Geography", "Seniority", "Gender"])

    if app_mode == " ":
            # # Render the readme as markdown using st.markdown.
    # readme_text = st.markdown(get_file_content_as_string("instructions.md"))
        st.markdown("# Time to start ")
        image = Image.open('utilities/banner.png')
        st.image(image)

        st.markdown(
            f"""
            Welcome to Investigators2020! Here you will find interactive datasets underlying the NHMRC Fellowship Outcomes across four key metrics - Research area, Geography, Seniority and Gender. You can explore how the Fellowship scheme has evolved with time and what characteristics are typical of successful applications within each funding tier.

            Each of the metrics can be accessed using the arrow to open the left sidebar, where you will also find instructions on how to interact with each of the datasets. In addition to various filter options, detailed data can be shown by hovering over individual datapoints and most plots can be interactively scaled by zooming in/out. The data is structured into funding tiers that align the pre- and post-2019 NHMRC schemes. For more details on how the data was collected and preprocessed, check out the resources below. If you have any other questions feel free to post an issue via the [GitHub repository](https://github.com/dezeraecox). Otherwise, happy exploring!

            <small>*Note: for comfortable viewing on mobile devices, rotating your devices to landscape is recommended!*</small>

            ## **Resources**

            - The original data was sourced from the [NHMRC website](https://www.nhmrc.gov.au/funding/data-research/outcomes-funding-rounds)

            - For more on the initial guidelines provided during the scheme restructure, check out the [NHMRC Factsheet](https://nhmrc.govcms.gov.au/about-us/resources/investigator-grants-2019-outcomes-factsheet)

            - Author track record information, including publication number and field-weighted citation impact, were collected from [SciVal](https://www.scival.com/). If you are considering an application in the upcoming round, it’s a great idea to benchmark yourself against previous successful applicants.

            - For more info on the specific number crunching and data visualisation techniques used here, check out my 2019 [Behind the scenes](https://github.com/dezeraecox/Behind-the-scenes---Investigator-Grants-2019) post.

            - For a summary of the scheme outcomes in 2020, and especially the outlook for ECRs, check out my recent article in [Research Professional News](https://www.researchprofessionalnews.com/rr-funding-insight-2020-9-emerging-researchers-face-uphill-struggle-at-nhmrc/)

            <small>*Disclaimer: the information contained here was intended to inform my personal decision of whether to apply for an Investigator Grant in the upcoming rounds.The information contained here is provided on an “as is” basis with no guarantees of completeness, accuracy, usefulness or timeliness. Any action you take as a result of this information is done so at your own peril. If you do decide to act on this information, however, I wish you the best of luck whichever path you may choose. May the odds be ever in your favour.*</small>
                
                    """, unsafe_allow_html=True
                )


    elif app_mode == "Research Area":
        logo_box.image(Image.open('utilities/banner.png'))

        st.sidebar.markdown("By default, all data is displayed for 2020. To explore individual research themes,  remove the ```All``` filter below and select filter(s) of interest then drag the slider to select a year.")

        area_filters = st.sidebar.multiselect("Which theme would you like to display?", ['All', 'Basic Science', 'Clinical Medicine and Science', 'Health Services Research', 'Public Health'], default='All')

        year = st.sidebar.select_slider("Which year would you like to display?", options=[2015, 2016, 2017, 2018, 2019, 2020], value=2020)

        run_areas(area_filters, year)

        st.sidebar.markdown('\n')
        st.sidebar.markdown('\n')

        st.sidebar.markdown("<small>*Note: Bubble plots are coloured according to most common association with the selected Broad Research Theme(s).*</small>", unsafe_allow_html=True,)
    

    elif app_mode == "Geography":
        logo_box.image(Image.open('utilities/banner.png'))

        st.sidebar.markdown("By default, success rate is displayed for 2020. To explore other metrics and years, select the filter of interest then drag the slider to select a year. Metrics according to specific research institutions are provided in a detailed table below.")

        metric_filter = st.sidebar.selectbox("Which metric would you like to display?", ['Success rate', 'Number of awards', 'Total funding ($M)'])

        year = st.sidebar.select_slider("Which year would you like to display?", options=[2015, 2016, 2017, 2018, 2019, 2020], value=2020)

        run_states(year, metric_filter)

    
    elif app_mode == "Seniority":
        logo_box.image(Image.open('utilities/banner.png'))

        st.sidebar.markdown("A summary of the seniority trends are shown in the first plot. To explore individual award levels more specifically, use the slider to select a year of interest.")

        year = st.sidebar.select_slider("Which year would you like to display?", options=[2015, 2016, 2017, 2018, 2019, 2020], value=2020)

        run_title(year)

        st.sidebar.markdown('\n')
        st.sidebar.markdown('\n')

        st.sidebar.markdown("<small>*Note: Publication metrics were collected with minimal filtering and data quality control to provide an overview of track record, as opposed to highlight specific individuals.*</small>", unsafe_allow_html=True,)
    

    elif app_mode == "Gender":
        logo_box.image(Image.open('utilities/banner.png'))

        st.sidebar.markdown("By default, all data is displayed. To explore individual combinations, remove the ```All``` filter below and select filters of interest.")

        gender_filters = st.sidebar.multiselect("Which gender would you like to display?", 
                         ['All', 'Male', 'Female', 'Not disclosed'], default=['All'])
        level_filters = st.sidebar.multiselect("Which level would you like to display?", 
                         ['All', 'Level 1', 'Level 2', 'Level 3'], default=['All'])
        run_gender(gender_filters, level_filters)

        st.sidebar.markdown("<small>*Individual data can be seen by hovering over the plots themselves, which can also be used to toggle datasets on or off by holding shift to select.*</small>", unsafe_allow_html=True,)

# ----------------------Define data elements----------------------

# To make Streamlit fast, st.cache allows us to reuse computation across runs.
# In this common pattern, we download data from an endpoint only once.
@st.cache(allow_output_mutation=True)
def load_data(data_file, source_path='data/'):
    return pd.read_csv(f'{source_path}{data_file}.csv')

def create_summary(df, filters, filter_col):
    summary = df.copy()
    summary = summary[summary[filter_col].isin(filters)]
    return summary

def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/streamlit/demo-self-driving/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


# ---------------------Define utility functions---------------------
class BubbleChart:
    """Borrowed class BubbleChart from https://matplotlib.org/devdocs/gallery/misc/packed_bubbles.html"""
    def __init__(self, area, bubble_spacing=0):
        """
        Setup for bubble collapse.

        Parameters
        ----------
        area : array-like
            Area of the bubbles.
        bubble_spacing : float, default: 0
            Minimal spacing between bubbles after collapsing.

        Notes
        -----
        If "area" is sorted, the results might look weird.
        """
        area = np.asarray(area)
        r = np.sqrt(area / np.pi)

        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        # calculate initial grid layout for bubbles
        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]

        self.com = self.center_of_mass()

    def center_of_mass(self):
        return np.average(
            self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3]
        )

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0],
                        bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - \
            bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        idx_min = np.argmin(distance)
        return idx_min if type(idx_min) == np.ndarray else [idx_min]

    def collapse(self, n_iterations=50):
        """
        Move bubbles to the center of mass.

        Parameters
        ----------
        n_iterations : int, default: 50
            Number of moves to perform.
        """
        for _i in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                # try to move directly towards the center of mass
                # direction vector from bubble to the center of mass
                dir_vec = self.com - self.bubbles[i, :2]

                # shorten direction vector to have length of 1
                dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                # calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # check whether new bubble collides with other bubbles
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # try to move around a bubble that you collide with
                    # find colliding bubble
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        # calculate direction vector
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                        # calculate orthagonal vector
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        # test which direction to go
                        new_point1 = (self.bubbles[i, :2] + orth *
                                      self.step_dist)
                        new_point2 = (self.bubbles[i, :2] - orth *
                                      self.step_dist)
                        dist1 = self.center_distance(
                            self.com, np.array([new_point1]))
                        dist2 = self.center_distance(
                            self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2


def bubbleplot(df, scale=5000, title=''):

    keywords = {
        'keywords': df['keyword'],
        'frequency': df['frequency'],
        'color': df['color']
        }

    bubble_chart = BubbleChart(area=keywords['frequency'],
                            bubble_spacing=0.1)
    bubble_chart.collapse()

    def area(r):
        return (np.pi * r**2)

    bubbles = pd.DataFrame(bubble_chart.bubbles)
    bubbles.columns = ['x', 'y', 'radius', 'area']
    bubbles[['x', 'y', 'radius']] = bubbles[['x', 'y', 'radius']] / 900
    bubbles['area'] = bubbles[['radius']].apply(area)
    bubbles['color'] = keywords['color']
    bubbles['keyword'] = keywords['keywords']
    bubbles['frequency'] = keywords['frequency']

    color_scheme = alt.Scale(
            domain=('#5d35f0', '#ab097d', '#cfa900', '#cf5d00'),
            range=['#5d35f0', '#ab097d', '#cfa900', '#cf5d00'])


    bubble_layer = alt.Chart(bubbles, title=title).mark_circle().encode(
        x=alt.X('x', scale=alt.Scale(domain=[bubbles['x'].min(), bubbles['x'].max()]), axis=None),
        y=alt.Y('y', scale=alt.Scale(domain=[bubbles['y'].min(), bubbles['y'].max()]), axis=None),
        size=alt.Size('area', scale=alt.Scale(range=[0, scale]), legend=None),
        color=alt.Color('color:N', scale=color_scheme, legend=None),
        tooltip=[alt.Tooltip('keyword:N', title='Keyword'),
        alt.Tooltip('frequency:Q', title='Frequency')]
        )

    annotation = alt.Chart(bubbles.sort_values(['frequency'], ascending=False)[0:5]).mark_text(
        align='center',
        fontSize = 10,
        limit=75
    ).encode(
        x=alt.X('x', scale=alt.Scale(domain=[bubbles['x'].min(), bubbles['x'].max()]), axis=None),
        y=alt.Y('y', scale=alt.Scale(domain=[bubbles['y'].min(), bubbles['y'].max()]), axis=None),
        text='keyword:N',
    )

    return (bubble_layer + annotation)

# -----------------------Define main pages-----------------------

def run_areas(area_filters=['All'], year=2020):

    if 'All' in area_filters:
        area_filters = ['Basic Science', 'Clinical Medicine and Science', 'Health Services Research', 'Public Health']

    # read in df
    df = load_data('research_area', source_path='data/')
    df.drop([col for col in df.columns.tolist()
             if 'Unnamed: ' in col], axis=1, inplace=True)

    # generate summary plot
    plot_area_summary(df)

    if len(area_filters) < 1:
        return

    # filter for year
    df = create_summary(df, filters=[year], filter_col='Year')
    df = create_summary(df, filters=area_filters, filter_col='Broad Research Area')

    plot_area_year(df, year, area_filters)


def run_states(year, metric_filter):

    # read in states df
    summary_df = load_data('states', source_path='data/')
    summary_df.drop([col for col in summary_df.columns.tolist()
             if 'Unnamed: ' in col], axis=1, inplace=True)

    summary_df = create_summary(summary_df, filters=[year], filter_col='Year')

    # read in locations df
    locations_df = load_data('locations', source_path='data/')
    locations_df.drop([col for col in locations_df.columns.tolist()
             if 'Unnamed: ' in col], axis=1, inplace=True)

    locations_df = create_summary(locations_df, filters=[year], filter_col='Year')
    
    plot_states(summary_df, locations_df, year, metric_filter)


def run_title(year=2020):

    # read in df
    df = load_data('seniority', source_path='data/')
    df.drop([col for col in df.columns.tolist()
             if 'Unnamed: ' in col], axis=1, inplace=True)

    # generate summary plot
    plot_title_summary(df)

    # filter for year
    df = create_summary(df, filters=[year], filter_col='Year')
    plot_title_year(df, year)

    pass


def run_gender(gender_filters=['All'], level_filters=['All']):

    # Generate plot

    if 'All' in gender_filters:
        gender_filters = ['Male', 'Female', 'Not disclosed']

    if 'All' in level_filters:
        level_filters = ['Level 1', 'Level 2', 'Level 3']

    filter_dict = {'Male': 'M', 'Female': 'F', 'Not disclosed': 'N', 'Level 1': 1.0, 'Level 2': 2.0, 'Level 3': 3.0, 'All': 'All'}
    gender_filters = [filter_dict[filter_name] for filter_name in gender_filters]
    level_filters = [filter_dict[filter_name] for filter_name in level_filters]

    # read in df
    df = load_data('gender', source_path='data/')
    df.drop([col for col in df.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)

    df = create_summary(df, filters=gender_filters, filter_col='gender')
    df = create_summary(df, filters=level_filters, filter_col='type_cat')

    # generate plot
    plot_gender(df)

    pass

# -----------------------Define plot elements-----------------------


def plot_gender(df):
    st.subheader('Funding success according to nominated gender')

    source = df.copy()
    source['id'] = source['gender'] + '_' + source['type_cat'].astype(int).astype(str)
    source['id'] = source['id'].map({'F_3': 'Female Level 3', 'F_2': 'Female Level 2', 'F_1': 'Female Level 1', 'M_3': 'Male Level 3', 'M_2': 'Male Level 2', 'M_1': 'Male Level 1', 'N_3': 'N.D. Level 3', 'N_2': 'N.D. Level 2', 'N_1': 'N.D. Level 1', })

    colour_scheme = alt.Scale(domain=('Female Level 3', 'Female Level 2', 'Female Level 1', 'Male Level 3', 'Male Level 2', 'Male Level 1', 'N.D. Level 3', 'N.D. Level 2', 'N.D. Level 1'), range=['#440063', '#852bad', '#c090d6', '#b35d12', '#de7f2c', '#f0b27d', '#404040', '#808080', '#b0b0b0'])
    
    selector = alt.selection_multi(empty='all', fields=['id'])

    base = alt.Chart(source).add_selection(selector)

    bars = base.mark_bar().encode(
        x=alt.X('year:O', axis=alt.Axis(title='Year')),
        y=alt.Y('sum(funded):Q', axis=alt.Axis(title='Number of grants funded')),
        color=alt.condition(selector, 'id:N', alt.value('lightgray'), scale=colour_scheme, legend=alt.Legend(title=' ')),
        tooltip=[alt.Tooltip('gender', title='Gender'), alt.Tooltip('type_cat', title='Funding Level'), alt.Tooltip('funded', title='Count')]
    ).interactive().properties(
    width=200,
    height=200,
    )

    total_amount = base.mark_line(point=True).encode(
        x=alt.X('year:O', axis=alt.Axis(title='Year')),
        y=alt.Y('amount:Q', axis=alt.Axis(title='Total funding ($M)')),
        color=alt.Color('id:N', scale=colour_scheme, legend=None),
        tooltip=[alt.Tooltip('gender', title='Gender'), alt.Tooltip('type_cat', title='Funding Level'), alt.Tooltip('amount', title='Amount')]
    ).transform_filter(
        selector
    ).interactive().properties(
    width=200,
    height=200,
    )

    proportion_total_funded = base.mark_line(point=True).encode(
        x=alt.X('year:O', axis=alt.Axis(title='Year')),
        y=alt.Y('proportion_total_funded:Q', axis=alt.Axis(title='Proportion of grants funded at level (%)')),
        color=alt.Color('id:N', scale=colour_scheme, legend=None),
        tooltip=[alt.Tooltip('gender', title='Gender'), alt.Tooltip('type_cat', title='Funding Level'), alt.Tooltip('proportion_total_funded', title='Proportion')]
    ).transform_filter(
        selector
    ).interactive().properties(
    width=200,
    height=200,
    )
    line = alt.Chart(pd.DataFrame({'y': [50]})).mark_rule(strokeDash=[10, 10]).encode(y='y')


    proportion_gender_funded = base.mark_line(point=True).encode(
        x=alt.X('year:O', axis=alt.Axis(title='Year')),
        y=alt.Y('proportion_gender_funded:Q', axis=alt.Axis(title='Success rate (%)')),
        color=alt.Color('id:N', scale=colour_scheme, legend=None),
        tooltip=[alt.Tooltip('gender', title='Gender'), alt.Tooltip('type_cat', title='Funding Level'), alt.Tooltip('proportion_gender_funded', title='Proportion')]
    ).transform_filter(
        selector
    ).interactive().properties(
    width=200,
    height=200,
    )

    # st.altair_chart((bars | total_amount | proportion_total_funded + line | proportion_gender_funded), use_container_width=True)
    st.altair_chart((bars| total_amount).configure_legend(
    orient='right'
    ), use_container_width=True)
    st.altair_chart(proportion_total_funded + line | proportion_gender_funded, use_container_width=True)


def plot_title_summary(df):

    st.subheader('Funding awarded according to academic seniority')

    # Prepare dataframe
    source = df.copy()
    summary_df = ((source.groupby(['Year', 'level']).count()['type_cat'] / source.groupby(['Year']).count()['type_cat']) * 100).reset_index()
    summary_df['title'] = summary_df['level'].map({0: 'Mr, Miss, Ms, Mrs', 1: 'Dr', 2: 'A/Pr', 3: 'Prof', 4: 'E/Pr'})

    # Colour schemes
    colour_scheme_titles = alt.Scale(domain=('Mr, Miss, Ms, Mrs', 'Dr', 'A/Pr', 'Prof', 'E/Pr'),
                              range=['#ffb5b7', '#ed666a', '#e62c31', '#bf0006', '#990005'])

    # Chart 1: Summary proportions
    summary_bars = alt.Chart(summary_df.reset_index()).mark_bar().encode(
        y='Year:N',
        x=alt.X('type_cat:Q', axis=alt.Axis(title='Proportion of awards')),
        order=alt.Order('level:O'),
        color=alt.Color('title:N', scale=colour_scheme_titles, sort=[
                        'Mr, Miss, Ms, Mrs', 'Dr', 'A/Pr', 'Prof', 'E/Pr'], legend=alt.Legend(title='Academic title')),
        tooltip=[alt.Tooltip('title', title='Academic title'), alt.Tooltip(
        'type_cat', title='Proportion of awards')]
        ).interactive().properties(width=900, height=200)

    st.altair_chart(summary_bars, use_container_width=True)


def plot_title_year(df, year):

    st.subheader(f'Awards according to academic seniority: {year}')

    # Prepare dataframe
    source = df.copy().sort_values(['Year', 'type_cat'], ascending=False)
    source['title'] = source['level'].map(
        {0: 'Mr, Miss, Ms, Mrs', 1: 'Dr', 2: 'A/Pr', 3: 'Prof', 4: 'E/Pr'})

    # Colour scheme
    colour_scheme_levels = alt.Scale(domain=(0, 1, 2, 3, 4),
                                     range=['#ffb5b7', '#ed666a', '#e62c31', '#bf0006', '#990005'])
    colour_scheme_titles = alt.Scale(domain=('Mr, Miss, Ms, Mrs', 'Dr', 'A/Pr', 'Prof', 'E/Pr'),
                                     range=['#ffb5b7', '#ed666a', '#e62c31', '#bf0006', '#990005'])

    # Chart elements
    proportion_df = (source.groupby(['title', 'level', 'type_cat']).count() / source.count() * 100)['CIA_title'].reset_index()
    proportion = alt.Chart(proportion_df).mark_circle().encode(
        x=alt.X('title:O', axis=alt.Axis(title='Academic title', labelAngle=0), sort=[
            'Mr, Miss, Ms, Mrs', 'Dr', 'A/Pr', 'Prof', 'E/Pr']),
        y=alt.Y('type_cat:O', axis=alt.Axis(title='Award level'), sort=[3, 2, 1]),
        color=alt.Color('title:O', scale=colour_scheme_titles, sort=[
            'Mr, Miss, Ms, Mrs', 'Dr', 'A/Pr', 'Prof', 'E/Pr'], legend=None),
        size=alt.Size('CIA_title', legend=None, scale=alt.Scale(range=[500, 5000])),
        tooltip=[alt.Tooltip('title', title='Academic Level'),
        alt.Tooltip('type_cat', title='Funding Level'), alt.Tooltip('CIA_title', title='Proportion', format='.2f')]
    ).interactive().properties(
        width=800,
        height=275,
    )

    fwci = altcat.catplot(source,
                height=200,
                mark='point',
                box_mark=dict(strokeWidth=2, opacity=0.6, color='lightgrey'),
                whisker_mark=dict(strokeWidth=2, opacity=0.9, color='lightgrey'),
                encoding=dict(
                            y=alt.Y('type_cat:O', title='Award level', sort=[3, 2, 1]),
                            x=alt.X('fwci_awarded:Q', title='Field-weighted citation index over ten years prior to award'),
                            color=alt.Color('level:N', legend=None, scale=colour_scheme_levels),
                            tooltip=[alt.Tooltip('title', title='Academic Level'),
                                    alt.Tooltip('type_cat', title='Funding Level'),
                                    alt.Tooltip('fwci_awarded', title='FWCI', format='.2f')]
                                ),
                transform='jitterbox',
                jitter_width=0.5)

    pubs = altcat.catplot(source,
                height=200,
                mark='point',
                box_mark=dict(strokeWidth=2, opacity=0.6, color='lightgrey'),
                whisker_mark=dict(strokeWidth=2, opacity=0.9, color='lightgrey'),
                encoding=dict(
                            y=alt.Y('type_cat:O', title='Award level', sort=[3, 2, 1]),
                            x=alt.X('pubs_awarded:Q', title='Publications awarded ten years prior to award'),
                            color=alt.Color('level:N', legend=None, scale=colour_scheme_levels),
                            tooltip=[alt.Tooltip('title', title='Academic Level'),
                                    alt.Tooltip('type_cat', title='Funding Level'),
                                    alt.Tooltip('pubs_awarded', title='Publications')]
                                ),
                transform='jitterbox',
                jitter_width=0.5)
    st.altair_chart(proportion, use_container_width=True)
    st.altair_chart(fwci.interactive(bind_y=False), use_container_width=True)
    st.altair_chart(pubs.interactive(bind_y=False), use_container_width=True)


def plot_area_summary(df):
    broad_proportion = ((df.groupby(['Year', 'Broad Research Area']).count(
    ) / df.groupby(['Year']).count())['Field of Research'])
    broad_proportion = broad_proportion * 100

    st.subheader('Funding awarded according to broad research theme')

    # Prepare dataframe
    source = broad_proportion.copy()

    # # Colour scheme
    # colour_scheme = alt.Scale(
    #     domain=('Basic Science', 'Clinical Medicine and Science', 'Health Services Research', 'Public Health'),
    #     range=['#5d35f0', '#ab097d', '#cfa900', '#cf5d00'])

    # # Chart 1: Summary proportions
    # bars = alt.Chart(source.reset_index()).mark_bar().encode(
    #     x=alt.X('Broad Research Area:N', axis=alt.Axis(labels=False, title=' ')),
    #     y=alt.Y('Field of Research:Q', axis=alt.Axis(title='Proportion of awards (%)')),
    #     color=alt.Color('Broad Research Area:N', scale=colour_scheme),
    #     column=alt.Column('Year:N', header=alt.Header(
    #         labelAngle=0,
    #         titleOrient='top',
    #         labelOrient='bottom',
    #         labelAlign='center',
    #         labelPadding=3,), title=' '
    #     ),
    #     tooltip=[alt.Tooltip('Broad Research Area:N', title='Research Area'), alt.Tooltip(
    #     'Field of Research:Q', title='Proportion of awards', format='.2f')]
    #     ).configure_view(stroke=None)

    # st.altair_chart(bars.properties(width=50, height=200), use_container_width=True)


    colour_scheme = {'Basic Science': '#5d35f0', 'Clinical Medicine and Science': '#ab097d', 'Health Services Research': '#cfa900', 'Public Health': '#cf5d00'}
    font = {'family' : 'Microsoft Tai Le',
    'weight' : 'normal',
    'size'   : 28 }
    matplotlib.rc('font', **font)
    plt.rcParams['svg.fonttype'] = 'none'
    fig, ax = plt.subplots(figsize=(20, 8))
    sns.barplot(
        x='Year',
        y='Field of Research',
        data=source.reset_index(),
        hue='Broad Research Area',
        palette=colour_scheme,
        ax=ax
    )
    ax.set_ylabel('Proportion of awards (%)')
    plt.legend(ncol=4, loc=9, prop={'size': 20})
    plt.tight_layout()
    plt.ylim(0, 55)

    st.pyplot(fig, use_container_width=True)

    pass


def plot_area_year(df, year, area_filters='All'):

    st.subheader(f'Common themes in {year}')

    area_colours = {'Basic Science': '#5d35f0', 'Clinical Medicine and Science':'#ab097d', 'Health Services Research':'#cfa900', 'Public Health': '#cf5d00'}

    source = df.copy()

    # ------assign most common color------
    color_counts = source.copy()
    color_counts['color'] = color_counts['Broad Research Area'].map(area_colours)
    color_counts['Field of Research'] = color_counts['Field of Research'].str.title().str.strip(' ')
    # Assign colour to most common theme i.e. if term appears most in Basic science versus clinical
    field_colors = color_counts.groupby(['Broad Research Area', 'Field of Research', 'color']).count().reset_index().sort_values(['Field of Research', 'code'], ascending=False).drop_duplicates(['Field of Research'])
    field_colors = dict(field_colors[['Field of Research', 'color']].values)

    kw_cols = [col for col in source.columns.tolist() if 'Res KW' in col]
    color_counts = pd.melt(source, id_vars=['Broad Research Area', 'Field of Research'], value_vars=kw_cols, value_name='keyword', var_name='color')
    color_counts['keyword'] = color_counts['keyword'].str.strip(' ')
    color_counts['color'] = color_counts['Broad Research Area'].map(area_colours)
    keyword_colors = color_counts.groupby(['Broad Research Area', 'keyword', 'color']).count().reset_index().sort_values(['keyword', 'Field of Research'], ascending=False).drop_duplicates(['keyword'])
    keyword_colors = dict(keyword_colors[['keyword', 'color']].values)

    # collect FOR by frequency, then shuffle to randomise circle position
    text = dict(pd.Series(source['Field of Research'].str.title().values.flatten()).str.strip(' ').value_counts())
    text_df = pd.DataFrame(list(text.values()), index=list(text.keys())).reset_index().rename(columns={0: 'frequency', 'index': 'keyword'}).sort_values('frequency', ascending=False)
    if len(text_df) > 50:
        sample = text_df.head(50).sample(frac=1).reset_index()
    else:
        sample = text_df.sample(frac=1).reset_index()
    sample['color'] = sample['keyword'].map(field_colors)

    for_bubbles = bubbleplot(sample, scale=4000, title='Fields of Research')

    # collect keywords by frequency, then shuffle to randomise circle position
    text = dict(pd.Series(source[kw_cols].values.flatten()).str.strip(' ').value_counts())
    text_df = pd.DataFrame(list(text.values()), index=list(text.keys())).reset_index().rename(columns={0: 'frequency', 'index': 'keyword'}).sort_values('frequency', ascending=False)
    sample = text_df.head(50).sample(frac=1).reset_index()
    sample['color'] = sample['keyword'].map(keyword_colors)


    kw_bubbles = bubbleplot(sample, scale=4000, title='Keywords')

    st.altair_chart((for_bubbles).configure_axis(grid=False).configure_view(strokeWidth=0), use_container_width=True)
    st.altair_chart((kw_bubbles).configure_axis(grid=False).configure_view(strokeWidth=0), use_container_width=True)


def plot_states(summary_df, locations_df, year, metric_filter='Success rate'):

    state_codes = { 'NSW': '1', 'VIC': '2', 'QLD': '3', 'SA': '4', 'WA': '5', 'TAS': '6', 'NT': '7', 'ACT': '8'
    }

    source = summary_df.copy()
    source.columns = ['Year', 'State and Territory', 'Applications', 'Number of awards', 'Success rate', 'Total funding ($M)']

    source['STATE_CODE'] = source['State and Territory'].map(state_codes)
    source['Success rate'] = source['Success rate'] * 100
    
    # remote geojson data object
    url_geojson = 'https://raw.githubusercontent.com/rowanhogan/australian-states/master/states.geojson'
    data_geojson_remote = alt.Data(url=url_geojson, format=alt.DataFormat(property='features',type='json'))

    # chart object
    map_plot = alt.Chart(data_geojson_remote).mark_geoshape(
    ).encode(
        color=alt.Color(f"{metric_filter}:Q", scale=alt.Scale(scheme='blues')),
        tooltip=[alt.Tooltip(f'{metric_filter}:Q', format='.1f')]
    ).properties(
        projection={'type': 'identity', 'reflectY': True},
        height=400,
        width=650
    ).transform_lookup(
        lookup='properties.STATE_CODE',
        from_=alt.LookupData(data=source, key='STATE_CODE',
                            fields=[f'{metric_filter}'])
    )

    st.altair_chart(map_plot, use_container_width=True)

    st.subheader(f'Total funding and awards in {year} by institution')
    
    # Generate top 5 institutions
    locations = locations_df.groupby('Admin Institution').agg(
    max_dollars=('Total', 'sum'), max_awards=('State', 'count')).rename(columns={'max_dollars': 'Total funding ($M)', 'max_awards': 'Awards'})
    locations['Total funding ($M)'] = locations['Total funding ($M)'] / 1000000
    st.table(locations.sort_values('Awards', ascending=False))


if __name__ == "__main__":
    main()

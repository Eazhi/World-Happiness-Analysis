import streamlit as st
import awesome_streamlit as ast

import src.pages.analysis
import src.pages.ml
import src.pages.home

# Disable deprecation warnings when calling st.pyplot() without a figure
st.set_option("deprecation.showPyplotGlobalUse", False)

st.title("World Happiness analysis")

PAGES = {
    "Home": src.pages.home,
    "Analysis": src.pages.analysis,
    "ML": src.pages.ml,
}


def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]

    with st.spinner(f"Loading {selection} ..."):
        ast.shared.components.write_page(page)


if __name__ == "__main__":
    main()

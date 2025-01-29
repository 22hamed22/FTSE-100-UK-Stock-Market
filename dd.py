import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
try:
    import tensorflow as tf
    st.write(f"TensorFlow version: {tf.__version__}")
except ImportError as e:
    st.write(f"Error importing TensorFlow: {e}")

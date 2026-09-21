from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_app_routes_real_input_and_validates_empty():
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'streamlit_app.py')).run(timeout=60)
    assert not app.exception
    app.text_input[0].set_value('what is my bank balance').run()
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert len(app.json)==1
    app.text_input[0].set_value(' ').run()
    app.button[0].click().run()
    assert len(app.warning)==1

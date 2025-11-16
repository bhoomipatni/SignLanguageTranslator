from flask import Flask, render_template

# Configure Flask to use the frontend folders for static files and templates
app = Flask(__name__, static_folder='frontend/static', template_folder='frontend/templates')


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
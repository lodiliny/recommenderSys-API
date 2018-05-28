from flask import Flask, request, jsonify, render_template, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_restless import APIManager
from flask_cors import CORS
import indexer
#from wtforms import Form, TextAreaField, validators

app3 = Flask(__name__)

app3.config['CORS_HEADERS'] = ["Content-Type", "Access-Control-Allow-Credentials"]
app3.config['CORS_RESOURCES'] ={r"/recSys/api/*": {"origins": "*"}}
app3.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///recSystem.db'

cors = CORS(app3, supports_credentials=True)
db = SQLAlchemy(app3)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sfu_id = db.Column(db.String(10), unique=True)
    first_name = db.Column(db.Text)
    last_name = db.Column(db.Text)

class Saved_Recommendations(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sfu_id = db.Column(db.String(20))
    rec_title = db.Column(db.Text)
    rec_url = db.Column(db.Text)
    rec_snippet = db.Column(db.Text)

class Recommendations(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sfu_id = db.Column(db.String(20))
    rec_title = db.Column(db.Text)
    rec_url = db.Column(db.Text)
    rec_snippet = db.Column(db.Text)
    rec_keyword = db.Column(db.Text)
    display_status = db.Column(db.Boolean)

class Highlights(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sfu_id = db.Column(db.String(20))
    highlights = db.Column(db.String(1000))
    tags = db.Column(db.String(500))
    batchNum = db.Column(db.Integer)
    rec_status = db.Column(db.Boolean)

'''class StudyAnswers(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sfu_id = db.Column(db.String(20))
    question1 = db.Column(db.String(10000))
    question2 = db.Column(db.String(10000))
    question3 = db.Column(db.String(10000))

class ReviewForm(Form):
    review1 = TextAreaField('Q1: ', [validators.DataRequired()])
    review2 = TextAreaField('Q2: ', [validators.DataRequired()])
    review3 = TextAreaField('Q3: ', [validators.DataRequired()])'''

@app3.route('/')
def welcome_page():
    return render_template("welcome_page.html")

@app3.route('/welcome', methods=['GET'])
def welcome():
    return render_template("index_page.html")

@app3.route('/question1', methods=['GET'])
def research_study1():
    #form = ReviewForm(request.form, csrf_enabled=False)
    return render_template("question.html")

@app3.route('/question2', methods=['GET'])
def research_study2():
    return render_template("study2.html")

@app3.route('/question3', methods=['GET'])
def research_study3():
    return render_template("study3.html")

@app3.route('/thankyou', methods=['GET'])
def thank_you():
    return render_template("thankyou_page.html")

@app3.route('/completed', methods=['GET'])
def completed():
    return redirect("https://www.surveymonkey.ca/r/recSys_feedback", code=302)

@app3.route('/demographics', methods=['GET'])
def demographics():
    return redirect("https://www.surveymonkey.ca/r/recSys_demographics", code=302)

@app3.route('/recSys/api/user', methods=['GET'])
def get_all_users():
    users = User.query.all()
    output = []
    for user in users:
        user_data = {}
        user_data['sfu_id'] = user.sfu_id
        user_data['first_name'] = user.first_name
        user_data['last_name'] = user.last_name
        output.append(user_data)
    return jsonify({'users': output})

@app3.route('/recSys/api/user/<sfu_id>', methods=['GET'])
def get_one_user(sfu_id):
    user = User.query.filter_by(sfu_id=sfu_id).first()
    if not user:
        return jsonify({'message': 'No user found'})
    user_data = {}
    user_data['sfu_id'] = user.sfu_id
    user_data['first_name'] = user.first_name
    user_data['last_name'] = user.last_name
    return jsonify({'user': user_data})

@app3.route('/recSys/api/user', methods=['POST'])
def create_user():
    data = request.get_json()
    new_user = User(sfu_id=data['sfu_id'], first_name=data['first_name'], last_name=data['last_name'])
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'New user created!'})

@app3.route('/recSys/api/highlight', methods=['POST'])
def add_highlights():
    data = request.get_json()
    userid = data['sfu_id']
    count = Highlights.query.filter_by(sfu_id=userid).count()
    print(count)
    if count == 0:
        new_highlight = Highlights(sfu_id=data['sfu_id'], highlights=data['Highlight'], tags=data['Tag'], batchNum=0, rec_status=False)
        db.session.add(new_highlight)
        db.session.commit()
    else:
        update_highlight = Highlights(sfu_id=data['sfu_id'], highlights=data['Highlight'], tags=data['Tag'], batchNum=count,
                                   rec_status=False)
        db.session.add(update_highlight)
        db.session.commit()
    #return jsonify({'message': 'New highlight added, recommendations ready!'})
    return 'New highlight added, recommendations ready!'

@app3.route('/recSys/api/compute-recommendations/<sfu_id>', methods=['GET'])
def compute_recommendation(sfu_id):
    user = Highlights.query.filter_by(sfu_id=sfu_id).all()
    output = []
    for entry in user:
        if entry.rec_status == False:
            highlights = entry.highlights
            if highlights == "":
                print("Contains no data")
            else:
                highlights = highlights.split(',')
                print(highlights)
                highlightlist = []
                for h in highlights:
                    highlightlist.append([h])
                print(highlightlist)
                recommendation = indexer.recsDetails(sfu_id, highlightlist)
                new_recommendation = Recommendations(sfu_id=recommendation['sfu_id'],
                                                     rec_title=recommendation['titles'], rec_url=recommendation['URLs'],
                                                     rec_snippet=recommendation['snippets'],
                                                     rec_keyword=recommendation['keywords'], display_status=False)
                db.session.add(new_recommendation)
                db.session.commit()

                entry.rec_status = True
                db.session.commit()
                output.append(recommendation)
    return jsonify({'Recommendations': output})

@app3.route('/recSys/api/display-recommendations/<sfu_id>', methods=['GET'])
def display_recommendations(sfu_id):
    user = Recommendations.query.filter_by(sfu_id=sfu_id).all()
    output = {}
    for rec in user:
        if rec.display_status == False:
            titles = rec.rec_title
            urls = rec.rec_url
            snippets = rec.rec_snippet
            keywords = rec.rec_keyword
            titles = titles.split(';,')
            output['title'] = titles
            urls = urls.split(';,')
            output['url'] = urls
            snippets = snippets.split(';,')
            output['snippet'] = snippets
            keywords = keywords.split(';,')
            output['keyword'] = keywords

            rec.display_status = True
            db.session.commit()

    return render_template("display.html", output=output)

@app3.route('/recSys/api/highlight/<sfu_id>', methods=['GET'])
def get_all_highlights_from_user(sfu_id):
    user = Highlights.query.filter_by(sfu_id=sfu_id).first()
    if not user:
        return jsonify({'message': "No user found"})
    user_data = {}
    user_data['sfu_id'] = user.sfu_id
    user_data['Highlights'] = user.highlights
    user_data['Tags'] = user.tags
    return jsonify({'User': user_data})

@app3.route('/recSys/api/users-highlights', methods=['GET'])
def get_all_users_highlights():
    users = Highlights.query.all()
    output = []
    for user in users:
        user_data = {}
        user_data['sfu_id'] = user.sfu_id
        user_data['Highlights'] = user.highlights
        user_data['Tags'] = user.tags
        output.append(user_data)
    return jsonify({'The Users and their highlights': output})

db.create_all()

api_manager = APIManager(app3, flask_sqlalchemy_db=db)
api_manager.create_api(User)

if __name__ == "__main__":
    app3.run(debug=True)
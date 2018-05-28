from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_restless import APIManager
from flask_cors import CORS
import indexer

app2 = Flask(__name__)

app2.config['CORS_HEADERS'] = ["Content-Type", "Access-Control-Allow-Credentials"]
app2.config['CORS_RESOURCES'] ={r"/recSys/api/*": {"origins": "*"}}
app2.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///people.db'

cors = CORS(app2, supports_credentials=True)
db = SQLAlchemy(app2)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sfu_id = db.Column(db.String(10), unique=True)
    first_name = db.Column(db.Text)
    last_name = db.Column(db.Text)

class Recommendations(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sfu_id = db.Column(db.String(20))
    recs = db.Column(db.String(20))

'''class Saved_Recommendations(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sfu_id = db.Column(db.String(20))
    rec_title = db.Column(db.Text)
    rec_url = db.Column(db.Text)
    rec_snippet = db.Column(db.Text)
    rec_keyword = db.Column(db.Text)'''

class Highlights(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sfu_id = db.Column(db.String(20))
    highlights = db.Column(db.String(1000))
    tags = db.Column(db.String(500))
    batchNum = db.Column(db.Integer)
    rec_status = db.Column(db.Boolean)

'''class HighlightList(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sfu_id = db.Column(db.String(10), unique=True)
    highlistlist = db.Column(ARRAY(db.JSON))'''

@app2.route('/')
def welcome_page():
    return "WELCOME TO THE HOME PAGE OF THE RECOMMENDER SYSTEM!"

@app2.route('/recSys/api/user', methods=['GET'])
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

@app2.route('/recSys/api/user/<sfu_id>', methods=['GET'])
def get_one_user(sfu_id):
    user = User.query.filter_by(sfu_id=sfu_id).first()
    if not user:
        return jsonify({'message': 'No user found'})
    user_data = {}
    user_data['sfu_id'] = user.sfu_id
    user_data['first_name'] = user.first_name
    user_data['last_name'] = user.last_name
    return jsonify({'user': user_data})

@app2.route('/recSys/api/user', methods=['POST'])
def create_user():
    data = request.get_json()
    new_user = User(sfu_id=data['sfu_id'], first_name=data['first_name'], last_name=data['last_name'])
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'New user created!'})

@app2.route('/recSys/api/highlight', methods=['POST'])
def add_highlights():
    data = request.get_json()
    userid = data['sfu_id']
    user = User.query.filter_by(sfu_id=userid)
    '''user_highlights = data['Highlight']
    user_highlights = user_highlights.split(',')
    compute_recommendation(userid, user_highlights)'''
    if not user:
        new_highlight = Highlights(sfu_id=data['sfu_id'], highlights=data['Highlight'], tags=data['Tag'], batchNum=0, rec_status=False)
        db.session.add(new_highlight)
        db.session.commit()
    else:
        num = user.batchNum + 1
        update_highlight = Highlights(sfu_id=data['sfu_id'], highlights=data['Highlight'], tags=data['Tag'], batchNum=num,
                                   rec_status=False)
        db.session.add(update_highlight)
        db.session.commit()
    #return jsonify({'message': 'New highlight added, recommendations ready!'})
    return 'New highlight added, recommendations ready!'

'''@app2.route('/recSys/api/display-recommendations/<sfu_id>', methods=['GET'])
def compute_recommendation(id):
    user = Highlights.query.filter_by(sfu_id=id)
    for entry in user:
        if entry.rec_status == False:
            highlights = entry.highlights
            highlights = highlights.split(';, ')
            highlightlist = []
            for h in highlights:
                highlightlist.append([h])
            recommendation = indexer.recsDetails(id, highlightlist)
            new_recommendation = Saved_Recommendations(sfu_id=recommendation['sfu_id'], rec_title=recommendation['titles'], rec_url=recommendation['URLs'], rec_snippet=recommendation['snippets'], rec_keyword=recommendation['keywords'])
            db.session.add(new_recommendation)
            db.session.commit()

            entry.rec_status = True
            db.session.commit()
    return jsonify({'message': 'Recommendations are ready!'}'''

@app2.route('/recSys/api/highlight/<sfu_id>', methods=['GET'])
def get_all_highlights_from_user(sfu_id):
    user = Highlights.query.filter_by(sfu_id=sfu_id).first()
    if not user:
        return jsonify({'message': "No user found"})
    user_data = {}
    user_data['sfu_id'] = user.sfu_id
    user_data['Highlights'] = user.highlights
    user_data['Tags'] = user.tags
    return jsonify({'User': user_data})

@app2.route('/recSys/api/users-highlights', methods=['GET'])
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

api_manager = APIManager(app2, flask_sqlalchemy_db=db)
api_manager.create_api(User)

if __name__ == "__main__":
    app2.run(debug=True)
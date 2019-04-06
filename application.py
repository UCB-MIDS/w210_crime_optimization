# W210 Police Deployment
# MACHINE LEARNING Microservices

import numpy as np
import pandas as pd
import tempfile
import pickle
import joblib
import json
import itertools
import configparser
import requests
from datetime import datetime
from collections import defaultdict
from flask import Flask
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy

application = Flask(__name__)
api = Api(application)
application.config.from_pyfile('config.py')
db = SQLAlchemy(application)

application.config['CORS_ENABLED'] = True
CORS(application)

## Define the DB model
class Community(db.Model):
    __tablename__ = 'community'
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.Integer)
    name = db.Column(db.String(255))

    def __str__(self):
        return self.name

class PoliceDistrict(db.Model):
    __tablename__ = 'policedistrict'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    address = db.Column(db.String(255))
    zipcode = db.Column(db.String(255))
    community = db.Column(db.Integer, db.ForeignKey('community.id'))
    community_rel = db.relationship('Community',backref=db.backref('policedistrict', lazy='joined'))
    patrols = db.Column(db.Integer)

    def __str__(self):
        return self.name

class Distance(db.Model):
    __tablename__ = 'distances'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    district = db.Column(db.Integer, db.ForeignKey('policedistrict.id'))
    district_rel = db.relationship('PoliceDistrict')
    community = db.Column(db.Integer, db.ForeignKey('community.id'))
    community_rel = db.relationship('Community')
    distance = db.Column(db.Float())

    def __str__(self):
        return self.id

class PatrolDeployment(db.Model):
    __tablename__ = 'patroldeployment'
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date)
    period = db.Column(db.String(255))
    district = db.Column(db.Integer, db.ForeignKey('policedistrict.id'))
    district_rel = db.relationship('PoliceDistrict')
    community = db.Column(db.Integer, db.ForeignKey('community.id'))
    community_rel = db.relationship('Community')
    patrols = db.Column(db.Integer())

    def __str__(self):
        return self.id

# Services to implement:
#   * Load deployment plan
#   * Deploy patrols
#   * Undeploy patrols
#   * Save deployment plan
#   * Optimize deployment plan

### In-Memory model
loaded = False
date = None
period = None
communities = None
policeDistricts = None
deployments = None
distances = None
crimecounts = None
mapCoverage = None
mapCrimes = None
mapDeploys = None
distanceCost = None
fairness = None

### Parameters

# Number of crimes one patrol can act upon in a shift
n_crimes_per_patrol = 6

# Number of patrols required to act on each type of crime
crimetype_weights = {
                        'THEFT': 1,
                        'SEXUAL ASSAULT': 1,
                        'NARCOTICS': 1,
                        'ASSAULT': 1,
                        'OTHER OFFENSE': 1,
                        'DECEPTIVE PRACTICE': 1,
                        'CRIMINAL TRESPASS': 1,
                        'WEAPONS VIOLATION': 1,
                        'PUBLIC INDECENCY': 1,
                        'OFFENSE INVOLVING CHILDREN': 1,
                        'PROSTITUTION': 1,
                        'INTERFERENCE WITH PUBLIC OFFICER': 1,
                        'HOMICIDE': 1,
                        'ARSON': 1,
                        'GAMBLING': 1,
                        'LIQUOR LAW VIOLATION': 1,
                        'KIDNAPPING': 1,
                        'STALKING': 1,
                        'NON - CRIMINAL': 1,
                        'HUMAN TRAFFICKING': 1,
                        'RITUALISM': 1,
                        'DOMESTIC VIOLENCE': 1
                     }

# Checks if the service is running
class checkService(Resource):
    def get(self):
        # Test if the service is up
        return {'message':'Planning and Optimization service is running.','result': 'success'}

# Checks if a deployment plan is already loaded
# If it is, return the deployment plan being worked on
class getLoadedDeploymentPlan(Resource):
    def get(self):
        # Check if there is already a deployment plan loaded and the date and time
        global loaded
        global date
        global period
        global communities
        global policeDistricts
        global distances
        global deployments
        global crimecounts
        global mapCoverage
        global mapCrimes
        global mapDeploys
        global distanceCost
        global fairness

        if not loaded:
            return {'message':'No Deployment Plan loaded.','result': 'failed'}
        else:
            return {'communities':communities,'districts':policeDistricts,'mapCoverage':mapCoverage,
                    'mapCrimes':mapCrimes,'mapDeploys':mapDeploys,'distanceCost':distanceCost,
                    'fairness':fairness,'result':'success'}

# Loads a deployment plan for a specific day and period
# Returns the data used by the dashboard
class loadDeploymentPlan(Resource):
    def get(self):

        global date
        global period
        global loaded
        global communities
        global policeDistricts
        global distances
        global deployments
        global crimecounts
        global mapCoverage
        global mapCrimes
        global mapDeploys
        global distanceCost
        global fairness
        global n_crimes_per_patrol
        global crimetype_weights

        # Get the passed date and period of the day
        argparser = reqparse.RequestParser()
        argparser.add_argument('date')
        argparser.add_argument('period')
        args = argparser.parse_args()

        if args['date'] is None:
            return {'message':'Missing date argument. Please pass a date to load the deployment.','result':'failed'}

        if args['period'] is None:
            return {'message':'Missing period argument. Supported periods: DAWN, MORNING, AFTERNOON, EVENING.','result':'failed'}

        # if args['period'] not in ['DAWN','MORNING','AFTERNOON','EVENING']:
        #     return {'message':'Invalid period argument. Supported periods: DAWN, MORNING, AFTERNOON, EVENING.','result':'failed'}

        # Get crime counts from the machine learning service
        date = datetime.strptime('04-05-2019','%m-%d-%Y').date()
        weekday = date.weekday()
        weekyear = date.isocalendar()[1]
        period = args['period']
        payload = {'weekday':json.dumps([weekday]),'weekyear':json.dumps([weekyear]),'hourday':json.dumps([period])}
        url ='http://localhost:60000/predict'
        r = requests.post(url, data=payload)
        crimepreds = r.json()['result']

        # Load the model from the database to the in-memory model
        communities = {}
        policeDistricts = {}
        deployments = {}
        distances = {}
        crimecounts = {}

        # Build the data to plot on the map
        mapCoverage = {}
        mapCrimes = {}
        mapDeploys = {}

        # Build the KPIs to draw on dashboard
        distanceCost = 0
        fairness = 0

        # Get the communities and zero crime counts and deployments
        for comm in db.session.query(Community):
            communities[comm.id] = {'id':comm.id,'code':comm.code,'name':comm.name}
            deployments[comm.id] = defaultdict(int)
            deployments[comm.id]['total'] = 0
            crimecounts[communities[comm.id]['code']] = {'absolute_count':0,'weighted_count':0}
            mapCoverage[communities[comm.id]['code']] = 0
            mapCrimes[communities[comm.id]['code']] = 0
            mapDeploys[communities[comm.id]['code']] = 0

        # Get distances between districts and communities
        for dist in db.session.query(Distance):
            distances[(dist.district,dist.community)] = dist.distance

        # Get crime predictions from arguments and calculate absolute and weighted count of crimes
        print(crimecounts)
        for pred in crimepreds:
            if (pred['communityArea'] is not None) and (pred['primaryType'] is not None) and (pred['communityArea'] != '0'):
                crimecounts[int(pred['communityArea'])]['absolute_count'] += pred['pred']
                crimecounts[int(pred['communityArea'])]['weighted_count'] += pred['pred']*crimetype_weights[pred['primaryType']]
                mapCrimes[communities[int(pred['communityArea'])]['code']] = crimecounts[communities[int(pred['communityArea'])]['code']]['absolute_count']

        # Get the police districts
        for pd in db.session.query(PoliceDistrict):
            policeDistricts[pd.id] = {'id':pd.id,'name':pd.name,'total_patrols':pd.patrols,'available_patrols':pd.patrols,'deployed_patrols':0}

        # Get deployments, calculating crime statistics for the map
        for deploy in db.session.query(PatrolDeployment)\
                                .filter(PatrolDeployment.date==args['date'])\
                                .filter(PatrolDeployment.period==args['period']):
            deployments[deploy.community][deploy.district] += deploy.patrols
            deployments[deploy.community]['total'] += deploy.patrols
            policeDistricts[deploy.district]['available_patrols'] -= deploy.patrols
            policeDistricts[deploy.district]['deployed_patrols'] += deploy.patrols
            if crimecounts[communities[deploy.community]['code']]['weighted_count'] != 0:
                mapCoverage[communities[deploy.community]['code']] = (deployments[deploy.community]['total']*n_crimes_per_patrol)/ \
                                                                     crimecounts[communities[deploy.community]['code']]['weighted_count']*100
            else:
                mapCoverage[communities[deploy.community]['code']] = 0
            mapDeploys[communities[deploy.community]['code']] = deployments[deploy.community]['total']
            distanceCost += deploy.patrols*distances[(deploy.district,deploy.community)]

        # Set the deployment plan as loaded
        loaded = True

        # Return data to the dashboard
        return {'communities':communities,'districts':policeDistricts,'mapCoverage':mapCoverage,
                'mapCrimes':mapCrimes,'mapDeploys':mapDeploys,'distanceCost':distanceCost,
                'fairness':fairness,'result':'success'}

# Deploys patrols from a district to a community, recalculates the KPIs and returns the data to update the dashboard
class deployPatrols(Resource):
    def get(self):

        global communities
        global policeDistricts
        global distances
        global deployments
        global crimecounts
        global mapCoverage
        global mapCrimes
        global mapDeploys
        global distanceCost
        global fairness
        global n_crimes_per_patrol
        global crimetype_weights

        # Check if a deployment plan has already been loaded
        if (deployments is None) or (communities is None) or (policeDistricts is None) or (distances is None) or not loaded:
            return {'message':'No deployment plan loaded','result':'failed'}

        # Get the passed district, community and number of patrols
        argparser = reqparse.RequestParser()
        argparser.add_argument('district')
        argparser.add_argument('community')
        argparser.add_argument('patrols')
        args = argparser.parse_args()

        # Check arguments
        if (args['district'] is None) or (args['community'] is None) or (args['patrols'] is None):
            return {'message':'Argument missing. Expected arguments: district, community, patrols.','result':'failed'}

        if policeDistricts[int(args['district'])]['available_patrols'] < int(args['patrols']):
            return {'message':'Number of patrols to be deployed is higher than number of available patrols.','result':'failed'}

        # Update the numbers of deployed patrols
        deployments[int(args['community'])][int(args['district'])] += int(args['patrols'])
        deployments[int(args['community'])]['total'] += int(args['patrols'])
        policeDistricts[int(args['district'])]['available_patrols'] -= int(args['patrols'])
        policeDistricts[int(args['district'])]['deployed_patrols'] += int(args['patrols'])

        # Recalculate coverage of crimes and other KPIs for the dashboard
        if crimecounts[communities[int(args['community'])]['code']]['weighted_count'] != 0:
            mapCoverage[communities[int(args['community'])]['code']] = (deployments[int(args['community'])]['total']*n_crimes_per_patrol)/ \
                                                                       crimecounts[communities[int(args['community'])]['code']]['weighted_count']*100
        else:
            mapCoverage[communities[int(args['community'])]['code']] = 0
        mapDeploys[communities[int(args['community'])]['code']] = deployments[int(args['community'])]['total']
        distanceCost += int(args['patrols'])*distances[(int(args['district']),int(args['community']))]

        # Return data to the dashboard
        return {'communities':communities,'districts':policeDistricts,'mapCoverage':mapCoverage,
                'mapCrimes':mapCrimes,'mapDeploys':mapDeploys,'distanceCost':distanceCost,
                'fairness':fairness,'result':'success'}

class undeployPatrols(Resource):
    def get(self):

        global communities
        global policeDistricts
        global distances
        global deployments
        global crimecounts
        global mapCoverage
        global mapCrimes
        global mapDeploys
        global distanceCost
        global fairness
        global n_crimes_per_patrol
        global crimetype_weights

        if (deployments is None) or (communities is None) or (policeDistricts is None) or (distances is None) or not loaded:
            return {'message':'No deployment plan loaded','result':'failed'}

        # Get the passed district, community and number of patrols
        argparser = reqparse.RequestParser()
        argparser.add_argument('district')
        argparser.add_argument('community')
        argparser.add_argument('patrols')
        args = argparser.parse_args()

        if (args['district'] is None) or (args['community'] is None) or (args['patrols'] is None):
            return {'message':'Argument missing. Expected arguments: district, community, patrols.','result':'failed'}

        if deployments[int(args['community'])][int(args['district'])] < int(args['patrols']):
            return {'message':'Number of patrols deployed is lower than number of patrols to be removed.','result':'failed'}

        deployments[int(args['community'])][int(args['district'])] -= int(args['patrols'])
        deployments[int(args['community'])]['total'] -= int(args['patrols'])
        policeDistricts[int(args['district'])]['available_patrols'] += int(args['patrols'])
        policeDistricts[int(args['district'])]['deployed_patrols'] -= int(args['patrols'])

        # Recalculate coverage of crimes and other KPIs for the dashboard
        if crimecounts[communities[int(args['community'])]['code']]['weighted_count'] != 0:
            mapCoverage[communities[int(args['community'])]['code']] = (deployments[int(args['community'])]['total']*n_crimes_per_patrol)/ \
                                                                       crimecounts[communities[int(args['community'])]['code']]['weighted_count']*100
        else:
            mapCoverage[communities[int(args['community'])]['code']] = 0
        mapDeploys[communities[int(args['community'])]['code']] = deployments[int(args['community'])]['total']
        distanceCost += int(args['patrols'])*distances[(int(args['district']),int(args['community']))]

        # Return data to the dashboard
        return {'communities':communities,'districts':policeDistricts,'mapCoverage':mapCoverage,
                'mapCrimes':mapCrimes,'mapDeploys':mapDeploys,'distanceCost':distanceCost,
                'fairness':fairness,'result':'success'}

# Persists the deployment plan on the database
class saveDeploymentPlan(Resource):
    def get(self):

        global date
        global period
        global deployments

        if (deployments is None) or (communities is None) or (policeDistricts is None) or (distances is None) or not loaded:
            return {'message':'No deployment plan loaded.','result':'failed'}

        # Get all previous entries from the deployment plan
        for deploy in db.session.query(PatrolDeployment)\
                                .filter(PatrolDeployment.date==date)\
                                .filter(PatrolDeployment.period==period):
            db.session.delete(deploy)

        # Input new entries from the loaded deployment plan
        for comm in deployments:
            for pd in deployments[comm]:
                if (pd != 'total'):
                    db.session.add(PatrolDeployment(date=date,period=period,community=comm,district=pd,patrols=deployments[comm][pd]))

        db.session.commit()

        return {'message':'Deployment plan saved succesfully.','result':'success'}

# Set API resources and endpoints
api.add_resource(checkService, '/')
api.add_resource(getLoadedDeploymentPlan,'/getLoadedDeploymentPlan')
api.add_resource(loadDeploymentPlan, '/loadDeploymentPlan')
api.add_resource(deployPatrols, '/deployPatrols')
api.add_resource(undeployPatrols, '/undeployPatrols')
api.add_resource(saveDeploymentPlan, '/saveDeploymentPlan')

if __name__ == '__main__':
    application.run(debug=True, port=61000)

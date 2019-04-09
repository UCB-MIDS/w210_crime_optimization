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
import math
import s3fs
import configparser
from docplex.mp.model import Model
from docplex.mp.context import Context
from scipy.stats import t
from datetime import datetime
from collections import defaultdict
from flask import Flask
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS, cross_origin
from flask_sqlalchemy import SQLAlchemy

### Load configuration file
s3fs.S3FileSystem.read_timeout = 5184000  # one day
s3fs.S3FileSystem.connect_timeout = 5184000  # one day
s3 = s3fs.S3FileSystem(anon=False)
config_file = 'w210policedata/config/optimization.ini'
try:
    temp_file = tempfile.NamedTemporaryFile(delete=True)
    s3.get(config_file,temp_file.name)
    config = configparser.ConfigParser()
    config.read(temp_file.name)
except:
    print('Failed to load service configuration file.')
    print('Creating new file with default values.')
    config = configparser.ConfigParser()
    config['GENERAL'] = {'MLServiceEndpoint': 'http://localhost:60000'}
    temp_file = tempfile.NamedTemporaryFile(delete=True)
    with open(temp_file.name, 'w') as confs:
        config.write(confs)
    s3.put(temp_file.name,config_file)
    temp_file.close()
ml_endpoint = config['GENERAL']['MLServiceEndpoint']

### Load CPLEX configuration file and library
config_file = 'w210policedata/config/docloud_config.py'
try:
    s3.get(config_file,'docloud_config.py')
except:
    print('Failed to load DOCplexCloud config file. CPLEX Optimization will not be available.')

### Load Flask configuration file
config_file = 'w210policedata/config/config.py'
try:
    s3.get(config_file,'config.py')
except:
    print('Failed to load application configuration file!')

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
    ethnicity = db.Column(db.Integer)

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
totalCoverage = None
mapCoverage = None
mapCrimes = None
mapDeploys = None
distanceCost = None
fairness = None

### Parameters

# Number of crimes one patrol can act upon in a shift
n_crimes_per_patrol = 6

# Coverage threshold per community above and below which penalty is applied to optimization model
upper_penalty_threshold = 150
lower_penalty_threshold = 25

# Maximum coverage tolerated in any given community
max_coverage = 200

# Minimum coverage tolerated in any given community
min_coverage = 25

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

# Calculation of deployment Fairness
# Uses a difference of means test as described in https://link.springer.com/article/10.1007%2Fs10618-017-0506-1
def calculateFairnessTStat(communities, deploymentPlan, cpxMode=False):
    comm_count = {0: 0, 1: 0}
    deploy_count = {0: 0, 1: 0}

    for comm in deploymentPlan:
        if (communities[comm]['ethnicity'] == 0) or (communities[comm]['ethnicity'] == 1):
            comm_count[1] += 1
            deploy_count[1] += deploymentPlan[comm]['total']
        else:
            comm_count[0] += 1
            deploy_count[0] += deploymentPlan[comm]['total']

    df = comm_count[0]+comm_count[1]-2

    if not cpxMode:
        if (deploy_count[0] == 0) and (deploy_count[1] == 0):
            return 0, df

    means = {0: deploy_count[0]/comm_count[0], 1: deploy_count[1]/comm_count[1]}

    variances = {0: 0, 1: 0}

    for comm in deploymentPlan:
        if (communities[comm]['ethnicity'] == 0) or (communities[comm]['ethnicity'] == 1):
            variances[1] += (deploymentPlan[comm]['total']-means[1])**2
        else:
            variances[0] += (deploymentPlan[comm]['total']-means[0])**2

    variances = {0: variances[0]/(comm_count[0]-1), 1: variances[1]/(comm_count[1]-1)}

    sigma = ((((comm_count[0]-1)*(variances[0]**2))+((comm_count[1]-1)*(variances[1]**2)))/(comm_count[0]+comm_count[1]-2))**0.5

    t = (means[0]-means[1])/(sigma*(((1/comm_count[0])+(1/comm_count[1]))**0.5))

    return t, df

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
        global totalCoverage
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
                    'fairness':fairness,'totalCoverage':totalCoverage,'result':'success'}

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
        global totalCoverage
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
        date = datetime.strptime(json.loads(args['date']),'%m-%d-%Y').date()
        weekday = date.weekday()
        weekyear = date.isocalendar()[1]
        period = json.loads(args['period'])
        payload = {'weekday':json.dumps([weekday]),'weekyear':json.dumps([weekyear]),'hourday':json.dumps([period])}
        url =ml_endpoint+'/predict'
        try:
            r = requests.post(url, data=payload)
            crimepreds = r.json()['result']
        except:
            return {'message':'Error loading crime predictions from Machine Learning service.','result':'failed'}

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
        totalCoverage = 0
        distanceCost = 0
        fairness = 0

        # Get the communities and zero crime counts and deployments
        for comm in db.session.query(Community):
            communities[comm.id] = {'id':comm.id,'code':comm.code,'name':comm.name,'ethnicity':comm.ethnicity}
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
                mapCoverage[communities[deploy.community]['code']] = ((deployments[deploy.community]['total']*n_crimes_per_patrol)/ \
                                                                     crimecounts[communities[deploy.community]['code']]['weighted_count'])*100
            else:
                mapCoverage[communities[deploy.community]['code']] = 0
            mapDeploys[communities[deploy.community]['code']] = deployments[deploy.community]['total']
            distanceCost += deploy.patrols*distances[(deploy.district,deploy.community)]

        # Calculate the total coverage of Crimes
        totalcrimes = 0
        totaldeploys = 0
        for comm in deployments:
            totalcrimes += crimecounts[communities[comm]['code']]['weighted_count']
            totaldeploys += deployments[comm]['total']
        totalCoverage = ((totaldeploys*n_crimes_per_patrol)/totalcrimes)*100

        # Calculate the deployment Fairness
        # First calculate the t-statistic
        t_stat, df = calculateFairnessTStat(communities, deployments)
        # Now assign the p-value as the fairness
        fairness = (1 - t.cdf(abs(t_stat), df)) * 2
        # Convert to percentage value
        fairness = fairness * 100

        # Set the deployment plan as loaded
        loaded = True

        # Return data to the dashboard
        return {'communities':communities,'districts':policeDistricts,'mapCoverage':mapCoverage,
                'mapCrimes':mapCrimes,'mapDeploys':mapDeploys,'distanceCost':distanceCost,
                'fairness':fairness,'totalCoverage':totalCoverage,'result':'success'}

# Deploys patrols from a district to a community, recalculates the KPIs and returns the data to update the dashboard
class deployPatrols(Resource):
    def get(self):

        global communities
        global policeDistricts
        global distances
        global deployments
        global crimecounts
        global totalCoverage
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

        args['district'] = json.loads(args['district'])
        args['community'] = json.loads(args['community'])
        args['patrols'] = json.loads(args['patrols'])

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
            mapCoverage[communities[int(args['community'])]['code']] = ((deployments[int(args['community'])]['total']*n_crimes_per_patrol)/ \
                                                                       crimecounts[communities[int(args['community'])]['code']]['weighted_count'])*100
        else:
            mapCoverage[communities[int(args['community'])]['code']] = 0
        mapDeploys[communities[int(args['community'])]['code']] = deployments[int(args['community'])]['total']
        distanceCost += int(args['patrols'])*distances[(int(args['district']),int(args['community']))]

        # Calculate the total coverage of Crimes
        totalcrimes = 0
        totaldeploys = 0
        for comm in deployments:
            totalcrimes += crimecounts[communities[comm]['code']]['weighted_count']
            totaldeploys += deployments[comm]['total']
        totalCoverage = ((totaldeploys*n_crimes_per_patrol)/totalcrimes)*100

        # Calculate the deployment Fairness
        # First calculate the t-statistic
        t_stat, df = calculateFairnessTStat(communities, deployments)
        # Now assign the p-value as the fairness
        fairness = (1 - t.cdf(abs(t_stat), df)) * 2
        # Convert to percentage value
        fairness = fairness * 100

        # Return data to the dashboard
        return {'communities':communities,'districts':policeDistricts,'mapCoverage':mapCoverage,
                'mapCrimes':mapCrimes,'mapDeploys':mapDeploys,'distanceCost':distanceCost,
                'fairness':fairness,'totalCoverage':totalCoverage,'result':'success'}

class undeployPatrols(Resource):
    def get(self):

        global communities
        global policeDistricts
        global distances
        global deployments
        global crimecounts
        global totalCoverage
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

        args['district'] = json.loads(args['district'])
        args['community'] = json.loads(args['community'])
        args['patrols'] = json.loads(args['patrols'])

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
            mapCoverage[communities[int(args['community'])]['code']] = ((deployments[int(args['community'])]['total']*n_crimes_per_patrol)/ \
                                                                       crimecounts[communities[int(args['community'])]['code']]['weighted_count'])*100
        else:
            mapCoverage[communities[int(args['community'])]['code']] = 0
        mapDeploys[communities[int(args['community'])]['code']] = deployments[int(args['community'])]['total']
        distanceCost -= int(args['patrols'])*distances[(int(args['district']),int(args['community']))]

        # Calculate the total coverage of Crimes
        totalcrimes = 0
        totaldeploys = 0
        for comm in deployments:
            totalcrimes += crimecounts[communities[comm]['code']]['weighted_count']
            totaldeploys += deployments[comm]['total']
        totalCoverage = ((totaldeploys*n_crimes_per_patrol)/totalcrimes)*100

        # Calculate the deployment Fairness
        # First calculate the t-statistic
        t_stat, df = calculateFairnessTStat(communities, deployments)
        # Now assign the p-value as the fairness
        fairness = (1 - t.cdf(abs(t_stat), df)) * 2
        # Convert to percentage value
        fairness = fairness * 100

        # Return data to the dashboard
        return {'communities':communities,'districts':policeDistricts,'mapCoverage':mapCoverage,
                'mapCrimes':mapCrimes,'mapDeploys':mapDeploys,'distanceCost':distanceCost,
                'fairness':fairness,'totalCoverage':totalCoverage,'result':'success'}

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

# Runs the optimization model on the deployment plan
class runOptimization(Resource):
    def get(self):

        global communities
        global policeDistricts
        global distances
        global deployments
        global crimecounts
        global totalCoverage
        global mapCoverage
        global mapCrimes
        global mapDeploys
        global distanceCost
        global fairness
        global n_crimes_per_patrol
        global crimetype_weights

        if (deployments is None) or (communities is None) or (policeDistricts is None) or (distances is None) or not loaded:
            return {'message':'No deployment plan loaded.','result':'failed'}

        # Get the passed arguments
        argparser = reqparse.RequestParser()
        argparser.add_argument('useFairness')
        args = argparser.parse_args()

        if json.loads(args['useFairness']) == 'yes':
            useFairness = True
        else:
            useFairness = False

        print("Building model...")
        # Create model
        context = Context.make_default_context()
        model = Model(name="CrimeDeployment", context=context)

        print("Creating variables...")
        # Create model variables
        # Assignment of patrols from each district to each community is a separate variable
        deployments_cpx = {}
        deployed_cpx = defaultdict(int)
        ethnicity_count = {0: 0, 1: 0}
        for community in communities:
            comm_id = communities[community]['id']
            deployments_cpx[comm_id] = {'total': 0}
            if (communities[community]['ethnicity'] == 0) or (communities[community]['ethnicity'] == 1):
                ethnicity_count[1] += 1
            else:
                ethnicity_count[0] += 1
            for district in policeDistricts:
                dist_id = policeDistricts[district]['id']
                max_patrols = policeDistricts[district]['total_patrols']
                deployments_cpx[comm_id][dist_id] = model.integer_var(lb=0, ub=max_patrols, name="c"+str(comm_id)+"d"+str(dist_id))
                deployed_cpx[dist_id] += deployments_cpx[comm_id][dist_id]
                deployments_cpx[comm_id]['total'] += deployments_cpx[comm_id][dist_id]

        print("Creating constraints...")
        # Add model constraints
        # Sum of deployed units over all communities can't be higher than number of available patrols
        for district in policeDistricts:
            dist_id = policeDistricts[district]['id']
            max_patrols = policeDistricts[district]['total_patrols']
            model.add_constraint(deployed_cpx[dist_id] <= max_patrols)
            model.add_constraint(deployed_cpx[dist_id] >= 0)

        print("Calculating objective...")
        print("     - Crime Coverage")
        # Calculate crime coverage in this deployment
        # We check the coverage, but also apply a penalty for coverages larger than a set threshold
        coverages = []
        totalDeployed = []
        totalCrimes = []
        penalties = []
        ethnicityDeployed = {0: [], 1: []}
        for community in communities:
            if crimecounts[communities[community]['code']]['weighted_count'] != 0:
                coverage = ((deployments_cpx[community]['total']*n_crimes_per_patrol)/crimecounts[communities[community]['code']]['weighted_count'])
                coverages.append(coverage)
                penalty = model.max(0,coverage-(upper_penalty_threshold/100))+model.max(0,(lower_penalty_threshold/100)-coverage)
                penalties.append(penalty)
                totalDeployed.append(deployments_cpx[community]['total']*n_crimes_per_patrol)
                totalCrimes.append(crimecounts[communities[community]['code']]['weighted_count'])
                if (communities[community]['ethnicity'] == 0) or (communities[community]['ethnicity'] == 1):
                    ethnicityDeployed[1].append(deployments_cpx[community]['total'])
                else:
                    ethnicityDeployed[0].append(deployments_cpx[community]['total'])

        avg_coverage = sum(coverages)/len(coverages)
        avg_penalty = sum(penalties)/len(penalties)
        citywideCoverage = sum(totalDeployed)/sum(totalCrimes)
        citywidePenalty = model.max(0,citywideCoverage-(upper_penalty_threshold/100))+model.max(0,(lower_penalty_threshold/100)-citywideCoverage)
        fairness = model.abs((sum(ethnicityDeployed[0])/ethnicity_count[0])-(sum(ethnicityDeployed[1])/ethnicity_count[1]))

        print("     - Distances")
        # Calculate the distances units have to drive to fulfill deployments
        distances_cpx = []
        for comm in deployments_cpx:
            for dist in deployments_cpx[comm]:
                if (dist != 'total'):
                    distance = distances[(dist,comm)]*deployments_cpx[comm][dist]
                    distances_cpx.append(distance)
        distanceCost_cpx = sum(distances_cpx)

        print("     - Combined Objective")
        # Now build the objective function
        obj = 1000000000*citywidePenalty-1000000000*citywideCoverage+1000000*avg_penalty-100000*avg_coverage+distanceCost_cpx
        if useFairness:
            print("     - Add Fairness Statistic to Combined Objective")
            # Add model objective
            # First calculate the fairness of the deployment
            obj = 10000*fairness+obj

        #model.add(maximize(obj))
        model.minimize(obj)

        print("Adding KPIs...")
        # Add the KPIs we're interested in tracking
        model.add_kpi(citywideCoverage, publish_name="City-wide Coverage")
        model.add_kpi(citywidePenalty, publish_name="City-wide Coverage Penalty")
        model.add_kpi(avg_coverage, publish_name="Avg. Coverage")
        model.add_kpi(avg_penalty, publish_name="Avg. Coverage Penalty")
        model.add_kpi(distanceCost_cpx, publish_name="Total Distance")
        if useFairness:
            model.add_kpi(fairness, publish_name="Fairness Simple Mean Difference")
        model.add_kpi(obj, publish_name="Combined Objective")

        print("Model ready:")
        model.print_information()

        print("Solving problem...")
        # Solve the problem
        msol = model.solve()

        print("Done! Getting results...")
        # Get result from solve
        # if msol.solve_details['status']:
        #     return {'solve_status':msol.get_solve_status(),'message':'Optimization executed but aborted due to exceeding maximum run time.','result':'failed'}

        # Get solution and place in deployments
        model.report()
        print(msol.solve_details)

        # First reset any plan existing in memory and its KPIs
        deployments = {}
        mapCoverage = {}
        mapDeploys = {}
        totalCoverage = 0
        distanceCost = 0
        fairness = 0
        for pd in db.session.query(PoliceDistrict):
            policeDistricts[pd.id] = {'id':pd.id,'name':pd.name,'total_patrols':pd.patrols,'available_patrols':pd.patrols,'deployed_patrols':0}

        for community in communities:
            comm_id = communities[community]['id']
            deployments[comm_id] = defaultdict(int)
            deployments[comm_id]['total'] = 0
            for district in policeDistricts:
                dist_id = policeDistricts[district]['id']
                n_patrols = msol['c'+str(comm_id)+'d'+str(dist_id)]
                deployments[comm_id][dist_id] += n_patrols
                deployments[comm_id]['total'] += n_patrols
                policeDistricts[dist_id]['available_patrols'] -= n_patrols
                policeDistricts[dist_id]['deployed_patrols'] += n_patrols
                if crimecounts[communities[comm_id]['code']]['weighted_count'] != 0:
                    mapCoverage[communities[comm_id]['code']] = ((deployments[comm_id]['total']*n_crimes_per_patrol)/ \
                                                                crimecounts[communities[comm_id]['code']]['weighted_count'])*100
                else:
                    mapCoverage[communities[comm_id]['code']] = 0
                mapDeploys[communities[comm_id]['code']] = deployments[comm_id]['total']
                distanceCost += n_patrols*distances[(dist_id,comm_id)]

        # Calculate the total coverage of Crimes
        totalcrimes = 0
        totaldeploys = 0
        for comm in deployments:
            totalcrimes += crimecounts[communities[comm]['code']]['weighted_count']
            totaldeploys += deployments[comm]['total']
            totalCoverage = ((totaldeploys*n_crimes_per_patrol)/totalcrimes)*100

        # Calculate the deployment Fairness
        # First calculate the t-statistic
        t_stat, df = calculateFairnessTStat(communities, deployments)
        # Now assign the p-value as the fairness
        fairness = (1 - t.cdf(abs(t_stat), df)) * 2
        # Convert to percentage value
        fairness = fairness * 100

        return {'communities':communities,'districts':policeDistricts,'mapCoverage':mapCoverage,
                'mapCrimes':mapCrimes,'mapDeploys':mapDeploys,'distanceCost':distanceCost,
                'fairness':fairness,'totalCoverage':totalCoverage,
                'solve_status':msol.solve_details.status,
                'message':'Optimization executed succesfully.','result':'success'}


# Set API resources and endpoints
api.add_resource(checkService, '/')
api.add_resource(getLoadedDeploymentPlan,'/getLoadedDeploymentPlan')
api.add_resource(loadDeploymentPlan, '/loadDeploymentPlan')
api.add_resource(deployPatrols, '/deployPatrols')
api.add_resource(undeployPatrols, '/undeployPatrols')
api.add_resource(saveDeploymentPlan, '/saveDeploymentPlan')
api.add_resource(runOptimization, '/runOptimization')

if __name__ == '__main__':
    application.run(debug=True, port=61000)

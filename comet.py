import pandas as pd
import numpy as np
from metaflow import FlowSpec, step, Parameter, current
from datetime import datetime
import os


assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'

os.environ['COMET_API_KEY'] = 'odKc3uuzuPK7Q28aXupHb9Yuz'
os.environ['MY_PROJECT_NAME'] = 'yt1420'

class Regression(FlowSpec):
 
    TEST_SPLIT = Parameter(
        name='test_split',
        help='Determining the split of the dataset for testing',
        default=0.20
    )

    @step
    def start(self):
        """
        Start up and print out some info to make sure everything is ok metaflow-side
        """
        print("Starting up at {}".format(datetime.utcnow()))
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)
        self.next(self.load_data)

    @step
    def load_data(self): 
        """
        Read the data in from the static file
        """

        self.data = pd.read_csv("credit_record.csv", index_col=0)
        self.df = pd.read_csv("application_record.csv", index_col=0)
        self.data.replace(('0','1','2','3','4','5'), (1,2,3,4,5,6), inplace=True)
        self.data.replace(('X', 'C'), (0, -1), inplace=True)
        self.status_avg = self.data.groupby('ID').mean()
        self.status_avg = self.status_avg.drop(columns = 'MONTHS_BALANCE')
        self.status_avg = self.status_avg.rename(columns = {'STATUS':'status_avg'})
        
    
        std = self.data.groupby('ID')['STATUS'].std(ddof=0)
        self.status_avg['std'] = std.tolist()

        self.data = pd.merge(self.data, self.status_avg, on='ID')
        self.data['target'] = np.where(self.data['status_avg']<0, 1, 0)

        self.num = ['CNT_CHILDREN','AMT_INCOME_TOTAL','DAYS_BIRTH','DAYS_EMPLOYED','CNT_FAM_MEMBERS']
        no_need_tochange = ['FLAG_WORK_PHONE','FLAG_PHONE','FLAG_EMAIL']
        replace = ['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY']
        get_dummies = ['NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE']

        self.numeric_features = self.num+no_need_tochange
        self.categorical_features = replace+get_dummies
        
        # go to the next step
        self.next(self.check_dataset)

    @step
    def check_dataset(self):
        for x in self.num:
            q_low = self.df[x].quantile(0.01)
            q_high = self.df[x].quantile(0.99)
            self.df = self.df[(self.df[x] >= q_low) & (self.df[x] <= q_high)]

        self.df1 = pd.DataFrame.merge(self.data, self.df, how ='inner', on = 'ID')
        self.df1=self.df1.drop(columns='FLAG_MOBIL')
        self.next(self.prepare_pipeline)

    @step
    def prepare_pipeline(self):
        from sklearn.model_selection import train_test_split

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),])

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),])
        
        from sklearn.compose import ColumnTransformer

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)])

        from sklearn.ensemble import RandomForestClassifier

        self.clf = Pipeline([
            ('preprocessor', preprocessor),
            ('estimator', RandomForestClassifier())])
        
        # with grid search
        # param_grid = {
        #     'estimator__max_depth': [80, 90],
        #     'estimator__n_estimators': [100, 200]}
        from sklearn.model_selection import GridSearchCV
        # self.grid_search = GridSearchCV(estimator = self.clf, param_grid = param_grid, 
        #                   cv = 3, n_jobs = -1, verbose = 2)

        X=self.df1[self.numeric_features+self.categorical_features]
        y=self.df1['target']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, 
                y, 
                stratify = y,
                test_size=0.3, 
                random_state=66)

        self.next(self.train_model)

    @step
    def train_model(self):
        """
        Train a regression on the training set
        """
        
        self.model =self.clf.fit(self.X_train, self.y_train)
        
        self.next(self.test_model)

    @step 
    def test_model(self):
        """
        Test the model on the hold out sample
        """
        from sklearn.metrics import classification_report
        from comet_ml import Experiment
        exp = Experiment(project_name=os.environ['MY_PROJECT_NAME'],auto_param_logging=False)

        y_predict = self.model.predict(self.X_test)
        y_train_predict=self.model.predict(self.X_train)
        print(classification_report(self.y_test,y_predict))
        print(classification_report(self.y_train,y_train_predict))

        from sklearn.metrics import f1_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import accuracy_score
        print(f1_score(self.y_test, y_predict))
        print(precision_score(self.y_test, y_predict))
        print(accuracy_score(self.y_test, y_predict))    

        self.params={
                    'random_state':66,
                    'model_type':'Random Forest Classifier',
                    'stratify':True}
        
        self.metrics={
                    'f1_score': f1_score(self.y_test, y_predict),
                    'precision_score': precision_score(self.y_test, y_predict),
                    'accuracy_score': accuracy_score(self.y_test, y_predict)}
        
        exp.log_parameters(self.params)
        exp.log_metrics(self.metrics)

        self.next(self.end)

    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))


if __name__ == '__main__':
    Regression()

import os
import sys 
import urllib.request
import pandas as pd
from calendar import monthrange
import pyspark
import pyspark.sql.functions as f
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import datetime
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s -  %(levelname)s-%(message)s')

def create_spark_session():
	conf = pyspark.SparkConf().setAppName('appName').setMaster('local')
	sc = pyspark.SparkContext(conf = conf)
	spark = SparkSession(sc)
	logging.info('******* Spark session created ******* ')
	return spark

def download_data(year, month):
	"""Function downloads data from http://data.gharchive.org for defined month and year.
	New directories for each day is created in current working directory and gzip data is downloaded there."""
	
	opener = urllib.request.URLopener()
	opener.addheader('User-Agent', 'whatever')

	cur_dir = os.getcwd()
	days = monthrange(int(year),int(month))[1]

	for d in range(1, days+1):
		date = '{}-{}-{}'.format(year, month, str(d).zfill(2))
		wd = '{}/{}'.format(cur_dir, date)
		if not os.path.isdir(wd):
			os.mkdir(wd)
		os.chdir(wd)

		for i in range(0, 24):
			url = 'http://data.gharchive.org/{}-{}.json.gz'.format(date, i)
			filename = '{}-{}.gz'.format(date, i)
			try:
				filename, headers = opener.retrieve(url, filename)
			except Exception as exc:
				logging.info('There was a problem for day %s hour %s: %s ' % (d, i, exc))
		logging.info('Data downloaded for day %s' % (d))
	logging.info('******* Data downloading ended. *******')
	
def filter_data(df):
	"""Function filters data frame choicing: created PullRequest Events, Issues Events, Fork Events and Star Events."""
	df = df.filter(((df.type=="PullRequestEvent") & (df.payload.action=='opened')) | \
		((df.type=="IssuesEvent") & (df.payload.action=='opened')) | \
		(df.type=="ForkEvent"))
	return df

def calculate_daily_date(df):
	"""Function creates additional column in data frame containing information about event's day"""
	df = df.withColumn("daily_date", df.created_at.substr(1 ,10))
	return df

def select_columns(df):
	"""Function select set of columns for further transformations and adds aliases"""
	df = df.selectExpr(["daily_date", "actor['id'] as actor_id", "actor['login'] as actor_login", \
						"repo['id'] as repo_id", "repo['name'] as repo_name", "type"]) 
	return df

def process_json(output_file, year, month, spark):
	"""Function processes json file for each day: 
	filtering data, calculating field, select columns and write results to parquet file."""
	
	days = monthrange(int(year),int(month))[1]
	for i in range(1, days+1):
		path="{}-{}-{}".format(year, month, str(i).zfill(2))
		df = spark.read.json(path)
		df = filter_data(df)
		df = calculate_daily_date(df)
		df = select_columns(df)
		df.write.parquet(output_file, mode='append', partitionBy=["daily_date"])
	logging.info('******* Github data processed successfully. *******')
	
def basic_agg(filename, spark):
	"""Function reads parquet file and creates basic aggregation view for further analysis."""
	
	df_full = spark.read.parquet(filename)
	df_agg = df_full.groupBy(["daily_date", "actor_id", "actor_login", "repo_id", "repo_name"]) \
		.pivot("type", values= ["ForkEvent", "PullRequestEvent", "IssuesEvent"]) \
		.agg(f.count("actor_id")) \
		.createOrReplaceTempView("aggreagation")
	logging.info('******* Basic aggregation done. *******')

def user_agg(output_file, spark):
	"""User aggregation calculates with use of inital aggreagation view.
	Results are written into parquet files."""
	
	spark.sql("""SELECT daily_date, actor_id, actor_login,
                 NVL(sum(PullRequestEvent),0) as PullRequestEvent_sum,
                 NVL(sum(IssuesEvent),0) as IssuesEvent_sum
                FROM aggreagation
                WHERE PullRequestEvent IS NOT NULL OR
                      IssuesEvent IS NOT NULL 
                GROUP BY daily_date, actor_id, actor_login
                ORDER BY daily_date, sum(PullRequestEvent) DESC""") \
	.write.parquet(output_file, mode='overwrite', partitionBy=["daily_date"])
	logging.info('******* User aggregation done. *******')

def repo_agg(output_file, spark):
	"""Repository aggregation calculates with use of inital aggreagation view.
	Results are written into parquet files."""
	
	spark.sql("""SELECT daily_date, repo_id, repo_name,
                 NVL(sum(PullRequestEvent),0) as PullRequestEvent_sum,
                 NVL(sum(IssuesEvent),0) as IssuesEvent_sum,
                 count(ForkEvent) as ForkEvent_count
                FROM aggreagation
                GROUP BY daily_date, repo_id, repo_name
                ORDER BY daily_date, sum(PullRequestEvent) DESC""") \
	.write.parquet(output_file, mode='overwrite', partitionBy=["daily_date"])
	logging.info('******* Repository aggregation done.*******')

def main():
	""" Run pipeline for given month and year in format 'YYYY', 'MM' """
	try:
		year = sys.argv[1]
		month = sys.argv[2]
		file = sys.argv[3]
		file_name = file + "_" + year + "_" + month + ".parquet"
		file_name_user = file + "_" + "user" + "_" + year + "_" + month + ".parquet"
		file_name_repo = file + "_" + "repo" + "_" + year + "_" + month + ".parquet"
	except:
		logging.info('Wrong parameters. Check if date is passed as year and month (YYYY, MM) and file name is passed.')
		
	if (int(year) >=2010) & (int(year)<= datetime.datetime.now().year) & (int(month)<=12) & (int(month)>=1) & (len(month)==2):
		spark = create_spark_session()
		download_data(year, month)
		process_json(file_name, year, month, spark)
		basic_agg(file_name, spark)
		user_agg(file_name_user, spark)
		repo_agg(file_name_repo, spark)
	else:
		logging.info('Wrong date. Check if date is passed as year and month (YYYY, MM)')
		
if __name__ == "__main__":
	main()
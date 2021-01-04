from cassandra.cluster import Cluster
from datetime import datetime, timedelta
import csv as csv
import pandas as pd
import math
from dateutil import parser
import os
import parameters as par
import numpy as np

keyspace = par.keyspace

def begin_session():

    # Create cluster.
    cluster = Cluster()

    # Connect to keyspace.
    session = cluster.connect(keyspace)

    return session;

def create_table(session):

    # Create table.
    session.execute("CREATE TABLE data (Time timestamp , Engine_Coolant_Temperature int , Calculated_Engine_Load_Value float , Absolute_Throttle_Position float , O2_Bank_1_Sensor_2_Short_Term_Fuel_Trim float, Intake_Air_Temperature int , O2_Bank_1_Sensor_1_Oxygen_Sensor_Voltage float, O2_Bank_1_Sensor_2_Oxygen_Sensor_Voltage float, Engine_RPM int , Intake_Manifold_Absolute_Pressure float , Timing_Advance float, Vehice_Speed int , O2_Bank_1_Sensor_1_Short_Term_Fuel_Trim_Bank_1 float, Short_Term_Fuel_Trim_Bank_1 float, Long_Term_Fuel_Trim_Bank_1 float, PRIMARY KEY(Time, Engine_Coolant_Temperature, Calculated_Engine_Load_Value, Absolute_Throttle_Position, Intake_Air_Temperature, Engine_RPM, Intake_Manifold_Absolute_Pressure, Vehice_Speed));")

class data_row:
    def __init__(self, time, engine_coolant_temperature, calculated_engine_load_value, absolute_throttle_position, intake_air_temperature, engine_rpm, intake_manifold_absolute_pressure, vehice_speed, long_term_fuel_trim_bank_1, o2_bank_1_sensor_1_oxygen_sensor_voltage, o2_bank_1_sensor_1_short_term_fuel_trim_bank_1, o2_bank_1_sensor_2_oxygen_sensor_voltage, o2_bank_1_sensor_2_short_term_fuel_trim, short_term_fuel_trim_bank_1, timing_advance):

        self.time                                                =   time
        self.engine_coolant_temperature                          =   engine_coolant_temperature
        self.calculated_engine_load_value                        =   calculated_engine_load_value
        self.absolute_throttle_position                          =   absolute_throttle_position
        self.intake_air_temperature                              =   intake_air_temperature
        self.engine_rpm                                          =   engine_rpm
        self.intake_manifold_absolute_pressure                   =   intake_manifold_absolute_pressure
        self.vehice_speed                                        =   vehice_speed
        self.long_term_fuel_trim_bank_1                          =   long_term_fuel_trim_bank_1
        self.o2_bank_1_sensor_1_oxygen_sensor_voltage            =   o2_bank_1_sensor_1_oxygen_sensor_voltage
        self.o2_bank_1_sensor_1_short_term_fuel_trim_bank_1      =   o2_bank_1_sensor_1_short_term_fuel_trim_bank_1
        self.o2_bank_1_sensor_2_oxygen_sensor_voltage            =   o2_bank_1_sensor_2_oxygen_sensor_voltage
        self.o2_bank_1_sensor_2_short_term_fuel_trim             =   o2_bank_1_sensor_2_short_term_fuel_trim
        self.short_term_fuel_trim_bank_1                         =   short_term_fuel_trim_bank_1
        self.timing_advance                                      =   timing_advance

        # 1000 Is a conversion factor from the datetime library timestamp to the cassandra timestamp.
        self.timestamp = round(datetime.timestamp(self.time)*1000)

    def add(self, session, table_name):

        # Add row to table.
        session.execute("INSERT INTO "+table_name+" (time, engine_coolant_temperature, calculated_engine_load_value, absolute_throttle_position, intake_air_temperature, engine_rpm, intake_manifold_absolute_pressure, vehice_speed, long_term_fuel_trim_bank_1, o2_bank_1_sensor_1_oxygen_sensor_voltage, o2_bank_1_sensor_1_short_term_fuel_trim_bank_1, o2_bank_1_sensor_2_oxygen_sensor_voltage, o2_bank_1_sensor_2_short_term_fuel_trim, short_term_fuel_trim_bank_1, timing_advance) VALUES ( "+str(self.timestamp)+","+str(self.engine_coolant_temperature)+","+str(self.calculated_engine_load_value)+","+str(self.absolute_throttle_position)+","+str(self.intake_air_temperature)+","+str(self.engine_rpm)+","+str(self.intake_manifold_absolute_pressure)+","+str(self.vehice_speed)+","+str(self.long_term_fuel_trim_bank_1)+","+str(self.o2_bank_1_sensor_1_oxygen_sensor_voltage)+","+str(self.o2_bank_1_sensor_1_short_term_fuel_trim_bank_1)+","+str(self.o2_bank_1_sensor_2_oxygen_sensor_voltage)+","+str(self.o2_bank_1_sensor_2_short_term_fuel_trim)+","+str(self.short_term_fuel_trim_bank_1)+","+str(self.timing_advance)+")");

def show_content(session):

    # Get session.
    rows = session.execute("SELECT * FROM test;")

    # Print all rows.
    for row in rows:
        print(row.time, row.engine_coolant_temperature, row.calculated_engine_load_value, row.absolute_throttle_position, row.intake_air_temperature, row.engine_rpm, row.intake_manifold_absolute_pressure, row.vehice_speed, row.long_term_fuel_trim_bank_1, row.o2_bank_1_sensor_1_oxygen_sensor_voltage, row.o2_bank_1_sensor_1_short_term_fuel_trim_bank_1, row.o2_bank_1_sensor_2_oxygen_sensor_voltage, row.o2_bank_1_sensor_2_short_term_fuel_trim, row.short_term_fuel_trim_bank_1, row.timing_advance)

    print("finished!");


def get_content_csv(filename, session, table_name):

    # Get content from cell that contains time.
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        row1 = next(reader)
        row2 = next(reader)

        # Convert content to datetime object.
        start_time_string = str(row2[0])
        start_time = parser.parse(start_time_string)

        print(start_time)

    # Import file and storage in dataframe.
    df = pd.read_csv (filename, skiprows = 2, decimal=',')

    # Delete rows with empty cells.
    df = df.dropna()

    # Show dataframe.
    print(df)

    # Put data in table.
    number_rows = df["Time (s)"].count()

    for i in range(number_rows):
        df.columns.get_loc("Time (s)")

        time = start_time
        time += timedelta(0,math.floor(df.iloc[i,df.columns.get_loc("Time (s)")]), (df.iloc[i,df.columns.get_loc("Time (s)")]-math.floor(df.iloc[i,df.columns.get_loc("Time (s)")]))*1000000)

        data_row_ = data_row(time, int(df.iloc[i,df.columns.get_loc("Engine Coolant Temperature (deg C)")]), df.iloc[i,df.columns.get_loc("Calculated Engine Load Value (%)")], df.iloc[i,df.columns.get_loc("Absolute Throttle Position (%)")], int(df.iloc[i,df.columns.get_loc("Intake Air Temperature (deg C)")]), int(df.iloc[i,df.columns.get_loc("Engine RPM (rpm)")]), df.iloc[i,df.columns.get_loc("Intake Manifold Absolute Pressure (kPa)")], int(df.iloc[i,df.columns.get_loc("Vehicle Speed (km/h)")]), df.iloc[i,df.columns.get_loc("Long Term Fuel Trim Bank 1 (%)")], df.iloc[i,df.columns.get_loc("O2 Bank 1 - Sensor 1 - Oxygen Sensor Voltage (V)")], df.iloc[i,df.columns.get_loc("O2 Bank 1 - Sensor 1 - Short Term Fuel Trim (%)")], df.iloc[i,df.columns.get_loc("O2 Bank 1 - Sensor 2 - Oxygen Sensor Voltage (V)")], df.iloc[i,df.columns.get_loc("O2 Bank 1 - Sensor 2 - Short Term Fuel Trim (%)")], df.iloc[i,df.columns.get_loc("Short Term Fuel Trim Bank 1 (%)")], df.iloc[i,df.columns.get_loc("Timing Advance for #1 cylinder (deg )")])

        data_row_.add(session, table_name);


def add_content_csv(session):

    # Get parameters
    path            = par.path
    start_directory = par.start_directory
    end_directory   = par.end_directory
    table_name      = par.table_name

    # Get session.
    session = begin_session()

    # Scanning directory for files and putting them in database.
    for entry in os.scandir(os.path.join(path, start_directory)):

        # Getting files name.
        filename_ = entry.name
        path_ = os.path.join(path, start_directory, filename_)

        # getting content from file and putting it in the database.
        get_content_csv(path_, session, table_name)

        # Move file to Processed directory.
        os.replace(os.path.join(path, start_directory, filename_), os.path.join(path, end_directory, filename_));

def get_content_database():

        # Get session.
        session = begin_session()

        # Get rows.
        rows = session.execute("SELECT * FROM test;")

        # Define column names.
        columns = np.array(["Time (s)", "Engine Coolant Temperature (deg C)", "Calculated Engine Load Value (%)", "Absolute Throttle Position (%)", "Intake Air Temperature (deg C)", "Engine RPM (rpm)", "Intake Manifold Absolute Pressure (kPa)", "Vehicle Speed (km/h)", "Long Term Fuel Trim Bank 1 (%)", "O2 Bank 1 - Sensor 1 - Oxygen Sensor Voltage (V)", "O2 Bank 1 - Sensor 1 - Short Term Fuel Trim (%)", "O2 Bank 1 - Sensor 2 - Oxygen Sensor Voltage (V)", "O2 Bank 1 - Sensor 2 - Short Term Fuel Trim (%)", "Short Term Fuel Trim Bank 1 (%)", "Timing Advance for #1 cylinder (deg )"])

        # Create dataframe.
        df = pd.DataFrame(columns = columns)

        # Set initial index.
        i = 0

        for row in rows:

            # Read row.
            data_array=[row.time, row.engine_coolant_temperature, row.calculated_engine_load_value, row.absolute_throttle_position, row.intake_air_temperature, row.engine_rpm, row.intake_manifold_absolute_pressure, row.vehice_speed, row.long_term_fuel_trim_bank_1, row.o2_bank_1_sensor_1_oxygen_sensor_voltage, row.o2_bank_1_sensor_1_short_term_fuel_trim_bank_1, row.o2_bank_1_sensor_2_oxygen_sensor_voltage, row.o2_bank_1_sensor_2_short_term_fuel_trim, row.short_term_fuel_trim_bank_1, row.timing_advance]

            # Update dataframe with new row.
            df.loc[i] = data_array

            # Update index.
            i += 1

        # Sort dataframe on time and reset the indices.
        df = df.sort_values(by=["Time (s)"])
        df = df.reset_index(drop=True)

        return df;

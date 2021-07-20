import regressiontree
import binset
import gdbt

def main():

    data = binset.build()
    
    gb = gdbt.GradientBoostingRegressor()
    gb.fit(data=data, n_estimators=2,
            learning_rate=0.5, max_depth=5, min_samples_split=2)

    print(gb.predict(data.train()))
    print(data.data)


if __name__ == "__main__":
    main()

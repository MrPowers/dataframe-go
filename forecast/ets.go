package forecast

import (
	"context"
	"errors"
	"fmt"

	"github.com/bradfitz/iter"
	"github.com/rocketlaunchr/dataframe-go"
)

// SesModel is a struct that holds necessary
// computed values for a forecasting result
type SesModel struct {
	data 		   *dataframe.SeriesFloat64
	testData       *dataframe.SeriesFloat64
	fcastData      *dataframe.SeriesFloat64
	initialLevel   float64
	originValue    float64
	smoothingLevel float64
	alpha          float64
	mae            float64
	sse            float64
	rmse           float64
	mape           float64
}

// SimpleExponentialSmoothing Function receives a series data of type dataframe.Seriesfloat64
// It returns a SesModel from which Fit and Predict method can be carried out.
func SimpleExponentialSmoothing(s *dataframe.SeriesFloat64) *SesModel {
	
	model := &SesModel{
		alpha:          0.0,
		data : 			&dataframe.SeriesFloat64{},
		testData:       &dataframe.SeriesFloat64{},
		fcastData:      &dataframe.SeriesFloat64{},
		initialLevel:   0.0,
		smoothingLevel: 0.0,
		mae:            0.0,
		sse:            0.0,
		rmse:           0.0,
		mape:           0.0,
	}

	model.data = s
	return model
}

// Fit Method performs the splitting and trainging of the SesModel based on the Exponential Smoothing algorithm.
// It returns a trained SesModel ready to carry out future predictions.
// The argument α must be between [0,1]. Recent values receive more weight when α is closer to 1.
func (sm *SesModel) Fit(ctx context.Context, α float64, r ...dataframe.Range ) (*SesModel, error) {
	
	if len(r) == 0 {
		r = append(r, dataframe.Range{})
	}

	count := len(sm.data.Values)
	if count == 0 {
		return nil, errors.New("no values in series range")
	}

	start, end, err := r[0].Limits(count)
	if err != nil {
		return nil, err
	}

	// Validation
	if end-start < 1 {
		return nil, errors.New("no values in series range")
	}

	if (α < 0.0) || (α > 1.0) {
		return nil, errors.New("α must be between [0,1]")
	}

	sm.alpha = α

	testData := sm.data.Values[end+1:]
	if len(testData) < 2 {
		return nil, errors.New("There should be a minimum of 2 data left as testing data")
	}

	testSeries := dataframe.NewSeriesFloat64("Test Data", nil)
	testSeries.Values = testData

	sm.testData = testSeries

	var st, Yorigin float64
	// Training smoothing Level
	for i := start; i < end+1; i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		xt := sm.data.Values[i]

		if i == start {
			st = xt
			sm.initialLevel = xt

		} else if i == end { // Setting the last value in traindata as Yorigin value for bootstrapping
			Yorigin = sm.data.Values[i]
			sm.originValue = Yorigin
		} else {
			st = α*xt + (1-α)*st
		}
	}
	sm.smoothingLevel = st

	fcast := []float64{}
	for k := end + 1; k < len(sm.data.Values); k++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		st = α * Yorigin + (1-α) * st
		fcast = append(fcast, st)

	}

	fcastSeries := dataframe.NewSeriesFloat64("Forecast Data", nil)
	fcastSeries.Values = fcast
	sm.fcastData = fcastSeries

	
	opts := &ErrorOptions{}

	mae, _, err := MeanAbsoluteError(ctx, testSeries, fcastSeries, opts)
	if err != nil {
		return nil, err
	}

	sse, _, err := SumOfSquaredErrors(ctx, testSeries, fcastSeries, opts)
	if err != nil {
		return nil, err
	}

	rmse, _, err := RootMeanSquaredError(ctx, testSeries, fcastSeries, opts)
	if err != nil {
		return nil, err
	}

	mape, _, err := MeanAbsolutePercentageError(ctx, testSeries, fcastSeries, opts)
	if err != nil {
		return nil, err
	}

	sm.sse = sse
	sm.mae = mae
	sm.rmse = rmse
	sm.mape = mape

	return sm, nil
}


// Predict method is used to run future predictions for Ses
// Using Ses Bootstrapping method
func (sm *SesModel) Predict(ctx context.Context, m int) (*dataframe.SeriesFloat64, error) {
	if m <= 0 {
		return nil, errors.New("m must be greater than 0")
	}

	forecast := make([]float64, 0, m)
	α := sm.alpha
	Yorigin := sm.originValue
	st := sm.smoothingLevel

	// Now calculate forecast
	for range iter.N(m) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		st = α*Yorigin + (1-α)*st
		forecast = append(forecast, st)
	}

	fdf := dataframe.NewSeriesFloat64("Prediction", nil)
	fdf.Values = forecast

	return fdf, nil
}

// Summary function is used to Print out Data Summary
// From the Trained Model
func (sm *SesModel) Summary() {

	alpha := dataframe.NewSeriesFloat64("Alpha", nil, sm.alpha)
	initLevel := dataframe.NewSeriesFloat64("Initial Level", nil, sm.initialLevel)
	st := dataframe.NewSeriesFloat64("Smooting Level", nil, sm.smoothingLevel)

	info := dataframe.NewDataFrame(alpha, initLevel, st)
	fmt.Println(info.Table())

	mae := dataframe.NewSeriesFloat64("MAE", nil, sm.mae)
	sse := dataframe.NewSeriesFloat64("SSE", nil, sm.sse)
	rmse := dataframe.NewSeriesFloat64("RMSE", nil, sm.rmse)
	mape := dataframe.NewSeriesFloat64("MAPE", nil, sm.mape)
	accuracyErrors := dataframe.NewDataFrame(sse, mae, rmse, mape)

	fmt.Println(accuracyErrors.Table())

	fmt.Println(sm.testData.Table())
	fmt.Println(sm.fcastData.Table())
}

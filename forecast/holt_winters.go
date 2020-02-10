package forecast

import (
	"context"
	"errors"
	"fmt"

	"github.com/bradfitz/iter"
	dataframe "github.com/rocketlaunchr/dataframe-go"
	pd "github.com/rocketlaunchr/dataframe-go/pandas"
)

// HwModel is a Model Interface that holds necessary
// computed values for a forecasting result
type HwModel struct {
	data                 *dataframe.SeriesFloat64
	trainData            *dataframe.SeriesFloat64
	testData             *dataframe.SeriesFloat64
	fcastData            *dataframe.SeriesFloat64
	initialSmooth        float64
	initialTrend         float64
	initialSeasonalComps []float64
	smoothingLevel       float64
	trendLevel           float64
	seasonalComps        []float64
	period               int
	alpha                float64
	beta                 float64
	gamma                float64
	errorM               *ErrorMeasurement
}

// HoltWinters Function receives a series data of type dataframe.Seriesfloat64
// It returns a HwModel from which Fit and Predict method can be carried out.
func HoltWinters(s *dataframe.SeriesFloat64) *HwModel {
	model := &HwModel{
		data:                 &dataframe.SeriesFloat64{},
		trainData:            &dataframe.SeriesFloat64{},
		testData:             &dataframe.SeriesFloat64{},
		fcastData:            &dataframe.SeriesFloat64{},
		initialSmooth:        0.0,
		initialTrend:         0.0,
		initialSeasonalComps: []float64{},
		smoothingLevel:       0.0,
		trendLevel:           0.0,
		seasonalComps:        []float64{},
		period:               0,
		alpha:                0.0,
		beta:                 0.0,
		gamma:                0.0,
		errorM:               &ErrorMeasurement{},
	}

	model.data = s
	return model
}

// Fit Method performs the splitting and trainging of the HwModel based on the Tripple Exponential Smoothing algorithm.
// It returns a trained HwModel ready to carry out future predictions.
// The arguments α, beta nd gamma must be between [0,1]. Recent values receive more weight when α is closer to 1.
func (hm *HwModel) Fit(ctx context.Context, o FitOptions) (*HwModel, error) {

	var (
		α, β, γ float64
		period  int
		r       *dataframe.Range
		errTyp  ErrorType
	)

	α = o.Alpha
	β = o.Beta
	γ = o.Gamma
	period = o.Period

	if o.TrainDataRange != nil {
		r = o.TrainDataRange
	}

	errTyp = o.ErrMtype

	start, end, err := r.Limits(len(hm.data.Values))
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

	if (β < 0.0) || (β > 1.0) {
		return nil, errors.New("β must be between [0,1]")
	}

	if (γ < 0.0) || (γ > 1.0) {
		return nil, errors.New("γ must be between [0,1]")
	}

	trainData := hm.data.Values[start : end+1]
	trainSeries := dataframe.NewSeriesFloat64("Train Data", nil, trainData)

	hm.trainData = trainSeries

	testData := hm.data.Values[end+1:]
	if len(testData) < 3 {
		return nil, errors.New("There should be a minimum of 3 data left as testing data")
	}

	testSeries := dataframe.NewSeriesFloat64("Test Data", nil)
	testSeries.Values = testData
	hm.testData = testSeries

	hm.alpha = α
	hm.beta = β
	hm.gamma = γ

	y := hm.data.Values[start : end+1]

	seasonals := initialSeasonalComponents(y, period)

	hm.initialSeasonalComps = initialSeasonalComponents(y, period)

	var trnd, prevTrnd float64
	trnd = initialTrend(y, period)
	hm.initialTrend = trnd

	var st, prevSt float64 // smooth

	for i := start; i < end+1; i++ {
		// Breaking out on context failure
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		xt := y[i]

		if i == start { // Set initial smooth
			st = xt

			hm.initialSmooth = xt

		} else {
			// multiplicative method
			// prevSt, st = st, α * (xt / seasonals[i % period]) + (1 - α) * (st + trnd)
			// prevTrnd, trnd = trnd, β * (st - prevSt) + (1 - β) * trnd
			// seasonals[i % period] = γ * (xt / (prevSt + prevTrnd)) + (1 - γ) * seasonals[i % period]

			// additive method
			prevSt, st = st, α*(xt-seasonals[i%period])+(1-α)*(st+trnd)
			prevTrnd, trnd = trnd, β*(st-prevSt)+(1-β)*trnd
			seasonals[i%period] = γ*(xt-prevSt-prevTrnd) + (1-γ)*seasonals[i%period]
			// _ = prevTrnd
			// fmt.Println(st + trnd + seasonals[i % period])
		}

	}

	// This is for the test forecast
	fcast := []float64{}
	m := 1
	for k := end + 1; k < len(hm.data.Values); k++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		// multiplicative Method
		// fcast = append(fcast, (st + float64(m)*trnd) * seasonals[(m-1) % period])

		// additive method
		fcast = append(fcast, (st+float64(m)*trnd)+seasonals[(m-1)%period])

		m++
	}

	fcastSeries := dataframe.NewSeriesFloat64("Forecast Data", nil)
	fcastSeries.Values = fcast
	hm.fcastData = fcastSeries

	hm.smoothingLevel = st
	hm.trendLevel = trnd
	hm.period = period
	hm.seasonalComps = seasonals

	// NOw to calculate the Errors
	opts := &ErrorOptions{}

	var val float64

	if errTyp == MAE {
		val, _, err = MeanAbsoluteError(ctx, testSeries, fcastSeries, opts)
		if err != nil {
			return nil, err
		}
	} else if errTyp == SSE {
		val, _, err = SumOfSquaredErrors(ctx, testSeries, fcastSeries, opts)
		if err != nil {
			return nil, err
		}
	} else if errTyp == RMSE {
		val, _, err = RootMeanSquaredError(ctx, testSeries, fcastSeries, opts)
		if err != nil {
			return nil, err
		}
	} else if errTyp == MAPE {
		val, _, err = MeanAbsolutePercentageError(ctx, testSeries, fcastSeries, opts)
		if err != nil {
			return nil, err
		}
	} else {
		return nil, errors.New("Unknown error type")
	}

	hm.errorM = &ErrorMeasurement{
		errorType: errTyp,
		value:     val,
	}

	return hm, nil
}

// Predict method runs future predictions for HoltWinter Model
// It returns result in dataframe.SeriesFloat64 format
func (hm *HwModel) Predict(ctx context.Context, h int) (*dataframe.SeriesFloat64, error) {

	// Validation
	if h <= 0 {
		return nil, errors.New("value of h must be greater than 0")
	}

	forecast := make([]float64, 0, h)

	st := hm.smoothingLevel
	seasonals := hm.seasonalComps
	trnd := hm.trendLevel
	period := hm.period

	m := 1
	for range iter.N(h) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		// multiplicative Method
		// fcast = append(fcast, (st + float64(m)*trnd) * seasonals[(m-1) % period])

		// additive method
		forecast = append(forecast, (st+float64(m)*trnd)+seasonals[(m-1)%period])

		m++
	}

	fdf := dataframe.NewSeriesFloat64("Prediction", nil)
	fdf.Values = forecast

	return fdf, nil
}

// Summary function is used to Print out Data Summary
// From the Trained Model
func (hm *HwModel) Summary() {

	alpha := dataframe.NewSeriesFloat64("Alpha", nil, hm.alpha)
	beta := dataframe.NewSeriesFloat64("Beta", nil, hm.beta)
	gamma := dataframe.NewSeriesFloat64("Gamma", nil, hm.gamma)
	period := dataframe.NewSeriesFloat64("Period", nil, hm.period)

	infoConstants := dataframe.NewDataFrame(alpha, beta, gamma, period)
	fmt.Println(infoConstants.Table())

	initSmooth := dataframe.NewSeriesFloat64("Initial Smooothing Level", nil, hm.initialSmooth)
	initTrend := dataframe.NewSeriesFloat64("Initial Trend Level", nil, hm.initialTrend)

	st := dataframe.NewSeriesFloat64("Smooting Level", nil, hm.smoothingLevel)
	trnd := dataframe.NewSeriesFloat64("Trend Level", nil, hm.trendLevel)

	infoComponents := dataframe.NewDataFrame(initSmooth, initTrend, st, trnd)
	fmt.Println(infoComponents.Table())

	initSeasonalComps := dataframe.NewSeriesFloat64("Initial Seasonal Components", nil)
	initSeasonalComps.Values = hm.initialSeasonalComps

	seasonalComps := dataframe.NewSeriesFloat64("Seasonal Components", nil)
	seasonalComps.Values = hm.seasonalComps

	seasonalComponents := dataframe.NewDataFrame(initSeasonalComps, seasonalComps)
	fmt.Println(seasonalComponents)

	errTyp := hm.errorM.Type()
	errVal := hm.errorM.Value()
	errorM := dataframe.NewSeriesFloat64(errTyp, nil, errVal)

	fmt.Println(errorM.Table())

	fmt.Println(hm.testData.Table())
	fmt.Println(hm.fcastData.Table())
}

// Describe outputs various statistical information of testData, trainData or mainData Series in SesModel
func (hm *HwModel) Describe(ctx context.Context, typ DataType, opts ...pd.DescribeOptions) {
	var o pd.DescribeOptions

	if len(opts) > 0 {
		o = opts[0]
	}

	data := &dataframe.SeriesFloat64{}

	if typ == TrainData {
		data = hm.trainData
	} else if typ == TestData {
		data = hm.testData
	} else if typ == MainData {
		data = hm.data
	} else {
		panic(errors.New("unrecognised data type selection specified"))
	}

	output, err := pd.Describe(ctx, data, o)
	if err != nil {
		panic(err)
	}
	fmt.Println(output)

	return
}

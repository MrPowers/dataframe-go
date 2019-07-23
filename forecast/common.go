// Copyright 2018 PJ Engineering and Business Solutions Pty. Ltd. All rights reserved.

package forecast

import (
	"context"
	"fmt"
	"strconv"

	"github.com/rocketlaunchr/dataframe-go"
)

// SumOfSquaredError is a function that calculates and returns
// the error measurement between actual data and predicted data
// generated from forecast algorithm
// s will be locked for the duration of the operation.
//
// Example:
//	ctx := context.Background()
//
//  s1 := dataframe.NewSeriesFloat64("price", nil, 445.43, 345.2, 565.56, 433.34, 585.23 ,593.32, 641.43)
//
//	numOfFcast := 5
// 	alpha := 0.453
//
//  forecastEts, err := forecast.SimpleExponentialSmoothing(ctx, s1, alpha, numOfFcast)
//  if err != nil {
//    panic(err)
//  }
//
// 	strtPt :=  len(s1.Values)-numOfFcast
//
//  fmt.Println(forecast.SumOfSquaredError(ctx, s1, forecastEts, dataframe.Range{Start: &strtPt}))
//  // Output: 28.824807
//
func SumOfSquaredError(ctx context.Context, actualM, forecastM *dataframe.SeriesFloat64, r ...dataframe.Range) float64 {

	if len(r) == 0 {
		r = append(r, dataframe.Range{})
	}

	// starting point marks the location where test data range starts
	// so a start range should already be defined in r
	// else all passed in data will be considered as test data
	start, _, err := r[0].Limits(len(actualM.Values))
	if err != nil {
		panic(err)
	}

	actual := actualM.Values[start:len(forecastM.Values)]

	length := len(actual)

	// trying to make sure forecast and actual data are of the same length
	forecast := forecastM.Values[:length]

	// Sum of Squared Deviation error
	sse := 0.0

	deviationErr := make([]float64, length)

	for i := 0; i < length; i++ {
		if err := ctx.Err(); err != nil {
			panic(err)
		}

		deviationErr[i] = actual[i] - forecast[i]
	}

	for _, elem := range deviationErr {
		sse += elem * elem
	}

	// formatting sse result into 4 decimal placess before return
	result, err := strconv.ParseFloat(fmt.Sprintf("%.6f", sse/100), 64)
	if err != nil {
		panic(err)
	}

	return result
}

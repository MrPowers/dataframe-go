package forecast

import (
	"fmt"
	"strconv"
	"time"

	dataframe "github.com/rocketlaunchr/dataframe-go"
)

// See: http://www.itl.nist.gov/div898/handbook/pmc/section4/pmc435.htm
func initialTrend(y []float64, period int) float64 {

	var sum float64
	sum = 0.0

	for i := 0; i < period; i++ {
		sum += (y[period+i] - y[i]) / float64(period)
	}

	return sum / float64(period)
}

// See: http://www.itl.nist.gov/div898/handbook/pmc/section4/pmc435.htm
func initialSeasonalComponents(y []float64, period int) []float64 {

	nSeasons := len(y) / period

	seasonalAverage := make([]float64, nSeasons)
	seasonalIndices := make([]float64, period)

	// computing seasonal averages
	for i := 0; i < nSeasons; i++ {
		for j := 0; j < period; j++ {
			seasonalAverage[i] += y[(i*period)+j]
		}
		seasonalAverage[i] /= float64(period)
	}

	// Calculating initial Seasonal component values

	for i := 0; i < period; i++ {
		for j := 0; j < nSeasons; j++ {
			// Multiplcative seasonal component
			// seasonalIndices[i] += y[(j*period)+i] / seasonalAverage[j]

			// Additive seasonal component
			seasonalIndices[i] += y[(j*period)+i] - seasonalAverage[j]
		}
		seasonalIndices[i] /= float64(nSeasons)
	}

	return seasonalIndices
}

// getDateTime function fetches time.Time data from dataframe in string format
// but converts and return string as time.Time
func getDateTime(data dataframe.Series, row int) (*time.Time, error) {
	var dateTimeValue string

	dateTimeValue = data.ValueString(row, dataframe.DontLock)
	dateTime, err := time.Parse(timeFormat, dateTimeValue)
	if err != nil {
		return nil, err
	}

	return &dateTime, nil
}

func interpolateMissingData(s dataframe.Series, row, column int) (interface{}, error) {

	var (
		result       interface{}
		previousData float64
		nxtData      float64
		err          error
	)

	v := s.ValueString(row-1, dataframe.DontLock)
	previousData, err = strconv.ParseFloat(v, 64)
	if err != nil {
		return nil, fmt.Errorf("can't force string to float64. row: %d column: %d", row-1, column)
	}

	if row+1 < s.NRows(dataframe.DontLock) {
		v = s.ValueString(row+1, dataframe.DontLock)
		nxtData, err = strconv.ParseFloat(v, 64)
		if err != nil {
			return nil, fmt.Errorf("can't force string to float64. row: %d column: %d", row-1, column)
		}

		result = (previousData + nxtData) / 2.0
	} else {
		result = previousData
	}

	return result, nil
}

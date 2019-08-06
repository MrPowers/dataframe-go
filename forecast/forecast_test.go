package forecast

import (
	"context"
	"fmt"
	"testing"

	"github.com/rocketlaunchr/dataframe-go"
)

func TestErrors(t *testing.T) {
	ctx := context.Background()
	init := &dataframe.SeriesInit{}

	data := dataframe.NewSeriesFloat64("Complete Data", nil, 445.43, 345.2, 565.56,
		433.34, 585.23, 593.32, 641.43, 654.35, 234.65, 567.45, 645.45, 445.34, 564.65,
		598.76, 676.54, 654.56, 564.76, 456.76, 656.57, 765.45)

	alpha := 0.53
	m := 10

	forecast, err := SimpleExponentialSmoothing(ctx, data, alpha, m, dataframe.Range{End: &[]int{14}[0]})
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}

	testData := data.Values[0:15] // This is exact subset of data fetched with range passed into forecast

	testSeries := dataframe.NewSeriesFloat64("Test SubData", init)

	// Load forecast data into series
	testSeries.Insert(testSeries.NRows(), testData[:])

	fmt.Println(data.Table())
	fmt.Println(testSeries.Table())
	fmt.Println(forecast.Table())

	opts := &ErrorOptions{}

	mae, nvalMae, err := MeanAbsoluteError(ctx, testSeries, forecast, opts)
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}

	sse, nvalSse, err := SumOfSquaredErrors(ctx, testSeries, forecast, opts)
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}

	rmse, nvalRmse, err := RootMeanSquaredError(ctx, testSeries, forecast, opts)
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}

	mape, nvalMape, err := MeanAbsolutePercentageError(ctx, testSeries, forecast, opts)
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}

	fmt.Println("alpha:", alpha)
	fmt.Printf("MAE: %f, nvalues: %d\n\n", mae, nvalMae)
	fmt.Printf("SSE: %f, nvalues: %d\n\n", sse, nvalSse)
	fmt.Printf("RMSE: %f, nvalues: %d\n\n", rmse, nvalRmse)
	fmt.Printf("MAPE: %f, nvalues: %d\n\n", mape, nvalMape)
}

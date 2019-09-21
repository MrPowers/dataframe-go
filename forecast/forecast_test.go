package forecast

import (
	"context"
	"fmt"
	"testing"

	"github.com/rocketlaunchr/dataframe-go"
)

func TestSes(t *testing.T) {
	ctx := context.Background()
	init := &dataframe.SeriesInit{}

	// data := dataframe.NewSeriesFloat64("Complete Data", nil, 445.43, 345.2, 565.56, 433.34, 585.23, 593.32, 641.43, 654.35, 234.65, 567.45, 645.45, 445.34, 564.65, 598.76, 676.54, 654.56, 564.76, 456.76, 656.57, 765.45, 755.43, 745.2, 665.56, 633.34, 585.23, 693.32, 741.43, 654.35, 734.65, 667.45, 545.45, 645.34, 754.65, 798.76, 776.54, 654.56, 664.76, 856.76, 776.57, 825.45, 815.43, 845.2, 765.56, 733.34, 785.23, 893.32, 841.43, 754.35, 524.65, 567.45, 715.45, 845.34, 864.65, 898.76, 876.54, 854.56, 864.76, 856.76, 726.57, 700.31, 815.43, 805.2, 855.56, 733.34, 785.23, 893.32, 641.43, 554.35, 734.63, 834.89)
	// m := 30
	alpha := 0.5

	data := dataframe.NewSeriesFloat64("simple data", nil, 1, 2, 3, 4, 5, 6, 7, 8, 9)
	m := 5

	forecast, err := SimpleExponentialSmoothing(ctx, data, alpha, m, dataframe.Range{End: &[]int{8}[0]})
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}

	//testData := data.Values[0:40] // This is exact subset of data fetched with range passed into forecast
	testData := data.Values[len(data.Values)-m:]
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

package forecast

import (
	"context"
	"fmt"
	"testing"

	"github.com/rocketlaunchr/dataframe-go"
)

func TestSes(t *testing.T) {
	ctx := context.Background()

	// data := dataframe.NewSeriesFloat64("Complete Data", nil, 445.43, 345.2, 565.56, 433.34, 585.23, 593.32, 641.43, 654.35, 234.65, 567.45, 645.45, 445.34, 564.65, 598.76, 676.54, 654.56, 564.76, 456.76, 656.57, 765.45, 755.43, 745.2, 665.56, 633.34, 585.23, 693.32, 741.43, 654.35, 734.65, 667.45, 545.45, 645.34, 754.65, 798.76, 776.54, 654.56, 664.76, 856.76, 776.57, 825.45, 815.43, 845.2, 765.56, 733.34, 785.23, 893.32, 841.43, 754.35, 524.65, 567.45, 715.45, 845.34, 864.65, 898.76, 876.54, 854.56, 864.76, 856.76, 726.57, 700.31, 815.43, 805.2, 855.56, 733.34, 785.23, 893.32, 641.43, 554.35, 734.63, 834.89)
	// m := 30
	alpha := 0.1

	data := dataframe.NewSeriesFloat64("simple data", nil, 1, 2, 3, 4, 5, 6, 7, 8, 9)
	m := 20

	fModel, err := SimpleExponentialSmoothing(ctx, data, alpha, dataframe.Range{End: &[]int{5}[0]})
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}
	//spew.Dump(fModel)

	fmt.Println(data.Table())

	fModel.Summary()

	fpredict, err := fModel.Predict(ctx, m)
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}

	fmt.Println(fpredict.Table())

	// fModel.Optimize()
}

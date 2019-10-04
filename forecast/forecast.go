package forecast

import (
	"context"

	"github.com/rocketlaunchr/dataframe-go"
)

// Model is an interface to group trained models of Different
// Algorithms in the Forecast package under similar generic standard
type Model interface {
	// Fit Method performs the splitting and training of the Model Interface based on the Forecast algorithm Implemented.
	// It returns a trained Model ready to carry out future forecasts.
	// The argument α must be between [0,1]. Recent values receive more weight when α is closer to 1.
	Fit(ctx context.Context, α float64, r ...dataframe.Range ) (Model, error)
	// Predict method is used to run future predictions for Ses
	// Using Bootstrapping method
	Predict(ctx context.Context, m int) (*dataframe.SeriesFloat64, error)
	// Summary function is used to Print out Data Summary
	// From the Trained Model
	Summary()
}
